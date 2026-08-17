import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { mkdtemp, mkdir, writeFile, rm } from 'fs/promises'
import { join } from 'path'
import { homedir, tmpdir } from 'os'
import zlib from 'zlib'

import { createDshProvider } from '../../src/providers/dsh.js'
import { calculateCost } from '../../src/models.js'
import type { ParsedProviderCall } from '../../src/providers/types.js'

// DSH session logs are concatenations of INDEPENDENT zstd frames (one per
// appended event batch), so fixtures must compress each batch separately —
// a single zstdCompressSync over the whole file is a different (single-frame)
// format than what DSH writes.

const zstdCompress = (zlib as { zstdCompressSync?: (buf: Buffer) => Buffer }).zstdCompressSync
const zstdUnavailable = !zstdCompress

let tmpDir: string

beforeEach(async () => {
  tmpDir = await mkdtemp(join(tmpdir(), 'dsh-test-'))
})

afterEach(async () => {
  await rm(tmpDir, { recursive: true, force: true })
})

function sessionHeader(opts: { id?: string; cwd?: string } = {}) {
  return JSON.stringify({
    type: 'session',
    version: 0,
    id: opts.id ?? 'session-00000000-0000-0000-0000-000000000001',
    createdAt: 1786707336131,
    cwd: opts.cwd ?? 'C:\\Users\\test\\myproject',
    delegationDepth: 0,
    agentPreset: 'cordis',
  })
}

function requestHeader(model: string, time = 1786707337000) {
  return JSON.stringify({
    type: 'request/header',
    seq: 10,
    time,
    data: { header: { config: { provider: 'deepseek-official', model, reasoningEffort: 'max', maxTokens: 256000 } } },
  })
}

function turnStart(turn: number, time: number) {
  return JSON.stringify({ type: 'turn/start', seq: 1, time, data: { turn } })
}

function userMessage(text: string, time: number) {
  return JSON.stringify({
    type: 'user/message',
    seq: 2,
    time,
    data: { content: [{ type: 'text', text }], source: { kind: 'user' }, role: 'user', id: 'msg-1' },
  })
}

function chunkUsage(turn: number, step: number, usage: Record<string, number>, time: number) {
  return JSON.stringify({
    type: 'assistant/chunk',
    seq: 3,
    time,
    data: { turn, step, chunk: { type: 'usage', usage } },
  })
}

function assistantMessage(turn: number, step: number, usage: Record<string, number> | undefined, time: number) {
  return JSON.stringify({
    type: 'assistant/message',
    seq: 4,
    time,
    data: {
      turn,
      step,
      message: { role: 'assistant', content: [{ type: 'text', text: 'done' }] },
      ...(usage ? { usage } : {}),
    },
  })
}

function toolCall(turn: number, step: number, name: string, args: Record<string, unknown>, time: number) {
  return JSON.stringify({
    type: 'tool/call',
    seq: 5,
    time,
    data: { turn, step, callId: `call_${name}`, name, arguments: JSON.stringify(args) },
  })
}

// Write one frame per batch of lines, matching DSH's append-per-batch layout.
async function writeZstdSession(projectDirName: string, sessionDirName: string, batches: string[][]) {
  const dir = join(tmpDir, 'sessions', projectDirName, sessionDirName)
  await mkdir(dir, { recursive: true })
  const filePath = join(dir, 'session.jsonl.zstd')
  const frames = batches.map(lines => zstdCompress!(Buffer.from(lines.join('\n') + '\n', 'utf-8')))
  await writeFile(filePath, Buffer.concat(frames))
  return filePath
}

async function writePlainSession(projectDirName: string, sessionDirName: string, lines: string[]) {
  const dir = join(tmpDir, 'sessions', projectDirName, sessionDirName)
  await mkdir(dir, { recursive: true })
  const filePath = join(dir, 'session.jsonl')
  await writeFile(filePath, lines.join('\n') + '\n')
  return filePath
}

async function parseAll(provider: ReturnType<typeof createDshProvider>, filePath: string): Promise<ParsedProviderCall[]> {
  const source = { path: filePath, project: 'myproject', provider: 'dsh' }
  const calls: ParsedProviderCall[] = []
  for await (const call of provider.createSessionParser(source, new Set()).parse()) {
    calls.push(call)
  }
  return calls
}

describe('dsh provider - session discovery', () => {
  it.skipIf(zstdUnavailable)('discovers a multi-frame zstd session, project from the header cwd', async () => {
    await writeZstdSession('--C-Users-test-myproject--', 'session-abc', [
      [sessionHeader({ cwd: 'C:\\Users\\test\\myproject' })],
      [assistantMessage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000)],
    ])

    const provider = createDshProvider(tmpDir)
    const sessions = await provider.discoverSessions()

    expect(sessions).toHaveLength(1)
    expect(sessions[0]!.provider).toBe('dsh')
    expect(sessions[0]!.project).toBe('myproject')
    expect(sessions[0]!.path).toContain('session.jsonl.zstd')
  })

  it('discovers the uncompressed session.jsonl variant (compression=none)', async () => {
    await writePlainSession('--home-u-proj--', 'session-plain', [
      sessionHeader({ cwd: '/home/u/proj' }),
      assistantMessage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000),
    ])

    const provider = createDshProvider(tmpDir)
    const sessions = await provider.discoverSessions()

    expect(sessions).toHaveLength(1)
    expect(sessions[0]!.path).toContain('session.jsonl')
    expect(sessions[0]!.path).not.toContain('zstd')
    expect(sessions[0]!.project).toBe('proj')
  })

  it('returns empty for a non-existent home', async () => {
    const provider = createDshProvider('/nonexistent/dsh/home')
    expect(await provider.discoverSessions()).toEqual([])
  })

  it('skips session dirs without a session log', async () => {
    await mkdir(join(tmpDir, 'sessions', '--x--', 'session-empty'), { recursive: true })
    const provider = createDshProvider(tmpDir)
    expect(await provider.discoverSessions()).toEqual([])
  })

  it('DSH_HOME relocates discovery; an empty string is treated as unset', async () => {
    const home = join(tmpDir, 'dsh-home')
    await mkdir(join(home, 'sessions', '--x--', 'session-env'), { recursive: true })
    await writeFile(
      join(home, 'sessions', '--x--', 'session-env', 'session.jsonl'),
      sessionHeader({ cwd: '/x' }) + '\n',
    )

    const saved = process.env['DSH_HOME']
    process.env['DSH_HOME'] = home
    try {
      const sessions = await createDshProvider().discoverSessions()
      expect(sessions).toHaveLength(1)
    } finally {
      if (saved === undefined) delete process.env['DSH_HOME']
      else process.env['DSH_HOME'] = saved
    }

    process.env['DSH_HOME'] = ''
    try {
      const roots = await createDshProvider().probeRoots!()
      expect(roots).toEqual([{ path: join(homedir(), '.dsh', 'sessions'), label: 'sessions' }])
    } finally {
      if (saved === undefined) delete process.env['DSH_HOME']
      else process.env['DSH_HOME'] = saved
    }
  })

  it('probeRoots reports the sessions dir under the factory root', async () => {
    expect(await createDshProvider('/tmp/dsh-a').probeRoots!()).toEqual([
      { path: join('/tmp/dsh-a', 'sessions'), label: 'sessions' },
    ])
  })
})

describe('dsh provider - parsing', () => {
  it.skipIf(zstdUnavailable)('decodes events spread across multiple independent zstd frames', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-multi', [
      [sessionHeader({ id: 'session-multi', cwd: 'C:\\Users\\test\\myproject' })],
      [turnStart(1, 1786707339000), userMessage('build the thing', 1786707339100)],
      [chunkUsage(1, 1, { inputTokens: 500, outputTokens: 50 }, 1786707340000)],
      [chunkUsage(1, 2, { inputTokens: 800, outputTokens: 80 }, 1786707341000)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(2)
    expect(calls[0]!.inputTokens).toBe(500)
    expect(calls[1]!.inputTokens).toBe(800)
  })

  it.skipIf(zstdUnavailable)('a final assistant/message usage REPLACES the earlier chunk sample for the same turn/step', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-replace', [
      [sessionHeader({ id: 'session-replace' })],
      [turnStart(1, 1786707339000)],
      // Early sample, then the final report of the SAME API call: the totals
      // must come from the final report only, not the sum of both.
      [chunkUsage(1, 1, { inputTokens: 14900, outputTokens: 600, reasoningTokens: 500 }, 1786707340000)],
      [assistantMessage(1, 1, { inputTokens: 14981, outputTokens: 656, cacheReadTokens: 0, reasoningTokens: 609 }, 1786707340050)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.inputTokens).toBe(14981)
    expect(calls[0]!.outputTokens).toBe(656)
    expect(calls[0]!.reasoningTokens).toBe(609)
    expect(calls[0]!.timestamp).toBe(new Date(1786707340050).toISOString())
  })

  it.skipIf(zstdUnavailable)('a chunk sample arriving after the final report does not overwrite it', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-late', [
      [sessionHeader({ id: 'session-late' })],
      [assistantMessage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340050)],
      [chunkUsage(1, 1, { inputTokens: 999, outputTokens: 99 }, 1786707340100)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.inputTokens).toBe(100)
  })

  it.skipIf(zstdUnavailable)('falls back to the chunk sample when no assistant/message usage arrives', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-sample', [
      [sessionHeader({ id: 'session-sample' })],
      [chunkUsage(2, 3, { inputTokens: 42, outputTokens: 7 }, 1786707340000)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.inputTokens).toBe(42)
    expect(calls[0]!.deduplicationKey).toBe('dsh:session-sample:2:3')
  })

  it.skipIf(zstdUnavailable)('steps inherit the model of the most recent request/header', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-model', [
      [sessionHeader({ id: 'session-model' })],
      [requestHeader('deepseek-v4-pro', 1786707337000)],
      [assistantMessage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000)],
      [assistantMessage(1, 2, { inputTokens: 200, outputTokens: 20 }, 1786707341000)],
      [requestHeader('deepseek-v4-flash', 1786707342000)],
      [assistantMessage(2, 1, { inputTokens: 300, outputTokens: 30 }, 1786707343000)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls.map(c => c.model)).toEqual(['deepseek-v4-pro', 'deepseek-v4-pro', 'deepseek-v4-flash'])
  })

  it.skipIf(zstdUnavailable)('bills reasoning tokens at the output rate', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-reason', [
      [sessionHeader({ id: 'session-reason' })],
      [requestHeader('deepseek-v4-pro')],
      [assistantMessage(1, 1, { inputTokens: 1000, outputTokens: 100, cacheWriteTokens: 50, cacheReadTokens: 500, reasoningTokens: 400 }, 1786707340000)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.costUSD).toBeCloseTo(calculateCost('deepseek-v4-pro', 1000, 500, 50, 500, 0), 12)
  })

  it.skipIf(zstdUnavailable)('collects mapped tools, skill names and bash commands from tool/call events', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-tools', [
      [sessionHeader({ id: 'session-tools' })],
      [
        toolCall(1, 1, 'read', { path: '/x/a.ts' }, 1786707339500),
        toolCall(1, 1, 'edit', { path: '/x/a.ts' }, 1786707339600),
        toolCall(1, 1, 'bash', { command: 'git status && bun test' }, 1786707339700),
        toolCall(1, 1, 'skill', { name: 'coding-agent-orchestration' }, 1786707339800),
        toolCall(1, 1, 'cordis_run', { id: 'j1' }, 1786707339900),
        chunkUsage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000),
      ],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.tools).toEqual(['Read', 'Edit', 'Bash', 'Skill', 'cordis_run'])
    expect(calls[0]!.bashCommands).toEqual(['git', 'bun'])
    expect(calls[0]!.skills).toEqual(['coding-agent-orchestration'])
  })

  it.skipIf(zstdUnavailable)('pairs the user message of the turn and carries session id and project', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-ctx', [
      [sessionHeader({ id: 'session-ctx', cwd: 'C:\\Users\\test\\myproject' })],
      [turnStart(1, 1786707339000), userMessage('first question', 1786707339100)],
      [chunkUsage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000)],
      [turnStart(2, 1786707350000), userMessage('second question', 1786707350100)],
      [chunkUsage(2, 1, { inputTokens: 200, outputTokens: 20 }, 1786707351000)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(2)
    expect(calls[0]!.userMessage).toBe('first question')
    expect(calls[1]!.userMessage).toBe('second question')
    expect(calls[0]!.sessionId).toBe('session-ctx')
    expect(calls[0]!.project).toBe('myproject')
    expect(calls[0]!.projectPath).toBe('C:\\Users\\test\\myproject')
  })

  it('parses the uncompressed session.jsonl variant', async () => {
    const filePath = await writePlainSession('--home-u-proj--', 'session-plain', [
      sessionHeader({ id: 'session-plain', cwd: '/home/u/proj' }),
      turnStart(1, 1786707339000),
      userMessage('hello', 1786707339100),
      chunkUsage(1, 1, { inputTokens: 123, outputTokens: 45 }, 1786707340000),
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.inputTokens).toBe(123)
    expect(calls[0]!.outputTokens).toBe(45)
  })

  it.skipIf(zstdUnavailable)('skips buckets whose usage is all zero', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-zero', [
      [sessionHeader({ id: 'session-zero' })],
      [assistantMessage(1, 1, { inputTokens: 0, outputTokens: 0 }, 1786707340000)],
    ])

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(0)
  })

  it.skipIf(zstdUnavailable)('ignores a torn final frame appended by a crashed writer', async () => {
    const dir = join(tmpDir, 'sessions', '--C-Users-test-myproject--', 'session-torn')
    await mkdir(dir, { recursive: true })
    const filePath = join(dir, 'session.jsonl.zstd')
    const good = zstdCompress!(Buffer.from(
      sessionHeader({ id: 'session-torn' }) + '\n' +
      chunkUsage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000) + '\n',
    ))
    const torn = zstdCompress!(Buffer.from(chunkUsage(1, 2, { inputTokens: 1, outputTokens: 1 }, 1786707341000) + '\n'))
    await writeFile(filePath, Buffer.concat([good, torn.subarray(0, Math.floor(torn.length / 2))]))

    const calls = await parseAll(createDshProvider(tmpDir), filePath)
    expect(calls).toHaveLength(1)
    expect(calls[0]!.inputTokens).toBe(100)
  })

  it.skipIf(zstdUnavailable)('deduplicates (turn, step) calls seen across multiple parses', async () => {
    const filePath = await writeZstdSession('--C-Users-test-myproject--', 'session-dedup', [
      [sessionHeader({ id: 'session-dedup' })],
      [chunkUsage(1, 1, { inputTokens: 100, outputTokens: 10 }, 1786707340000)],
    ])

    const provider = createDshProvider(tmpDir)
    const source = { path: filePath, project: 'myproject', provider: 'dsh' }
    const seenKeys = new Set<string>()

    const firstRun: ParsedProviderCall[] = []
    for await (const call of provider.createSessionParser(source, seenKeys).parse()) firstRun.push(call)
    const secondRun: ParsedProviderCall[] = []
    for await (const call of provider.createSessionParser(source, seenKeys).parse()) secondRun.push(call)

    expect(firstRun).toHaveLength(1)
    expect(secondRun).toHaveLength(0)
  })

  it('handles a missing session file gracefully', async () => {
    const provider = createDshProvider(tmpDir)
    const source = { path: join(tmpDir, 'nope', 'session.jsonl.zstd'), project: 'test', provider: 'dsh' }
    const calls: ParsedProviderCall[] = []
    for await (const call of provider.createSessionParser(source, new Set()).parse()) calls.push(call)
    expect(calls).toHaveLength(0)
  })
})

describe('dsh provider - display names', () => {
  const provider = createDshProvider('/tmp')

  it('has correct name and displayName', () => {
    expect(provider.name).toBe('dsh')
    expect(provider.displayName).toBe('DeepSeek Harness')
  })

  it('maps deepseek models to readable names and passes unknown ids through', () => {
    expect(provider.modelDisplayName('deepseek-v4-pro')).toBe('DeepSeek v4 Pro')
    expect(provider.modelDisplayName('some-future-model')).toBe('some-future-model')
  })

  it('normalizes tool names, keeping unknown names raw', () => {
    expect(provider.toolDisplayName('bash')).toBe('Bash')
    expect(provider.toolDisplayName('pwsh')).toBe('Bash')
    expect(provider.toolDisplayName('todo_write')).toBe('TodoWrite')
    expect(provider.toolDisplayName('cordis_run')).toBe('cordis_run')
  })
})
