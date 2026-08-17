import { open, readdir, readFile, stat } from 'fs/promises'
import { join } from 'path'
import { homedir } from 'os'
import zlib from 'zlib'

import { readSessionFile } from '../fs-utils.js'
import { calculateCost, getShortModelName } from '../models.js'
import { extractBashCommands } from '../bash-utils.js'
import type { ProbeRoot, Provider, SessionSource, SessionParser, ParsedProviderCall } from './types.js'

// DeepSeek Harness (dsh) stores one session per directory:
//   <DSH_HOME|~/.dsh>/sessions/<encoded-cwd>/session-<uuid>/session.jsonl.zstd
// (or an uncompressed session.jsonl when compression=none). The .zstd file is
// a concatenation of INDEPENDENT zstd frames — one per appended event batch —
// so node:zlib's one-shot zstdDecompressSync (which decodes a single frame)
// must be driven frame-by-frame behind a structural frame-boundary scan. The
// scan below is a port of scanZstdFrames from the official
// @deepseek-ai/dsh-session-persistence-jsonl package.

// zstd landed in node:zlib in 22.15 / 23.8; the package floor is lower, so the
// provider degrades with a notice instead of assuming the export exists.
const zstdDecompress = (zlib as { zstdDecompressSync?: (buf: Buffer) => Buffer }).zstdDecompressSync

const ZSTD_MAGIC = 0xfd2fb528
let warnedZstdUnavailable = false

function warnZstdUnavailable(): void {
  if (warnedZstdUnavailable) return
  warnedZstdUnavailable = true
  process.stderr.write('codeburn: DSH sessions need Node >= 22.15 (zstd support); skipping DSH usage.\n')
}

type ZstdFrame = { start: number; end: number }

// Locate complete frames without decompressing their blocks. An EOF inside the
// final frame (a torn append from a crashed writer) returns its start so the
// caller can ignore the tail; invalid complete structure rejects.
function scanZstdFrames(buffer: Buffer, maxFrames = Number.POSITIVE_INFINITY): { frames: ZstdFrame[]; tornStart?: number } {
  const frames: ZstdFrame[] = []
  let offset = 0
  while (offset < buffer.length) {
    const start = offset
    if (buffer.length - offset < 4) return { frames, tornStart: start }
    if (buffer.readUInt32LE(offset) !== ZSTD_MAGIC) {
      throw new Error(`invalid zstd frame magic at byte ${offset}`)
    }
    offset += 4
    if (offset === buffer.length) return { frames, tornStart: start }
    const descriptor = buffer.readUInt8(offset)!
    offset += 1
    if ((descriptor & 24) !== 0) throw new Error(`reserved frame-header bit at byte ${offset - 1}`)
    const contentSizeFlag = descriptor >>> 6
    const singleSegment = (descriptor & 32) !== 0
    const checksum = (descriptor & 4) !== 0
    const dictionaryFlag = descriptor & 3
    const dictionaryBytes = dictionaryFlag === 3 ? 4 : dictionaryFlag
    const contentSizeBytes = contentSizeFlag === 0 ? (singleSegment ? 1 : 0) : 1 << contentSizeFlag
    const remainingHeaderBytes = (singleSegment ? 0 : 1) + dictionaryBytes + contentSizeBytes
    if (buffer.length - offset < remainingHeaderBytes) return { frames, tornStart: start }
    offset += remainingHeaderBytes
    for (;;) {
      if (buffer.length - offset < 3) return { frames, tornStart: start }
      const blockHeader = buffer.readUIntLE(offset, 3)
      offset += 3
      const lastBlock = (blockHeader & 1) !== 0
      const blockType = (blockHeader >>> 1) & 3
      const blockSize = blockHeader >>> 3
      if (blockType === 3) throw new Error(`reserved block type at byte ${offset - 3}`)
      const payloadBytes = blockType === 1 ? 1 : blockSize
      if (buffer.length - offset < payloadBytes) return { frames, tornStart: start }
      offset += payloadBytes
      if (lastBlock) break
    }
    if (checksum) {
      if (buffer.length - offset < 4) return { frames, tornStart: start }
      offset += 4
    }
    frames.push({ start, end: offset })
    if (frames.length === maxFrames) return { frames }
  }
  return { frames }
}

type DshUsage = {
  inputTokens?: number
  outputTokens?: number
  cacheReadTokens?: number
  cacheWriteTokens?: number
  reasoningTokens?: number
}

type DshEvent = {
  type?: string
  seq?: number
  time?: number
  // Session header fields live at the top level of the first event.
  id?: string
  cwd?: string
  data?: {
    turn?: number
    step?: number
    content?: Array<{ type?: string; text?: string }>
    header?: { config?: { model?: string; provider?: string } }
    chunk?: { type?: string; usage?: DshUsage }
    usage?: DshUsage
    name?: string
    arguments?: string
  }
}

type StepBucket = {
  usage: DshUsage
  // A usage report from assistant/message is the final value for its
  // (turn, step) and replaces an earlier assistant/chunk sample (the two are
  // adjacent reports of the same API call, per dsh-token-meter's usage
  // projection). Time follows the winning report.
  final: boolean
  time?: number
  // Model in force when this step's usage was reported (the most recent
  // request/header config at that point in the log).
  model: string
  tools: string[]
  skills: string[]
  bashCommands: string[]
}

const toolNameMap: Record<string, string> = {
  bash: 'Bash',
  pwsh: 'Bash',
  read: 'Read',
  write: 'Write',
  edit: 'Edit',
  str_replace_editor: 'Edit',
  glob: 'Glob',
  grep: 'Grep',
  todo_write: 'TodoWrite',
  todo: 'TodoWrite',
  web_search: 'WebSearch',
  skill: 'Skill',
  agent: 'Agent',
  ask_user_question: 'AskUserQuestion',
}

function mapToolName(raw: string): string {
  return toolNameMap[raw] ?? raw
}

function getDshHome(override?: string): string {
  // An empty-string DSH_HOME is treated as unset.
  return override ?? (process.env['DSH_HOME'] || undefined) ?? join(homedir(), '.dsh')
}

// DSH writes native-platform paths into the header (backslashes on Windows);
// split on both separators so discovery is correct on any host.
function projectFromCwd(cwd: string, fallback: string): string {
  const segments = cwd.split(/[\\/]/).filter(Boolean)
  return segments[segments.length - 1] ?? fallback
}

// Decode every complete frame and yield its JSONL lines. A torn final frame is
// ignored; a structurally corrupt file throws for the caller to report.
function* readZstdLines(buffer: Buffer, maxFrames = Number.POSITIVE_INFINITY): Generator<string> {
  const { frames } = scanZstdFrames(buffer, maxFrames)
  for (const frame of frames) {
    const text = zstdDecompress!(buffer.subarray(frame.start, frame.end)).toString('utf-8')
    for (const line of text.split('\n')) {
      if (line.trim()) yield line
    }
  }
}

async function readEventLines(filePath: string): Promise<string[] | null> {
  if (filePath.endsWith('.zstd')) {
    if (!zstdDecompress) {
      warnZstdUnavailable()
      return null
    }
    let buffer: Buffer
    try {
      buffer = await readFile(filePath)
    } catch {
      return null
    }
    try {
      return [...readZstdLines(buffer)]
    } catch (err) {
      process.stderr.write(`codeburn: skipped corrupt DSH session log ${filePath}: ${err instanceof Error ? err.message : err}\n`)
      return null
    }
  }
  const content = await readSessionFile(filePath)
  if (content === null) return null
  return content.split('\n').filter(l => l.trim())
}

// Cheap discovery probe: decompress ONLY the first frame (the session header
// batch) instead of the whole log. The header frame is tiny, so a bounded head
// read almost always contains it; fall back to a full read when it does not.
async function readSessionHeader(filePath: string): Promise<DshEvent | null> {
  const firstLine = async (): Promise<string | null> => {
    if (filePath.endsWith('.zstd')) {
      if (!zstdDecompress) {
        warnZstdUnavailable()
        return null
      }
      let head: Buffer
      try {
        const handle = await open(filePath, 'r')
        try {
          const size = (await handle.stat()).size
          const length = Math.min(size, 256 * 1024)
          head = Buffer.alloc(length)
          await handle.read(head, 0, length, 0)
        } finally {
          await handle.close()
        }
      } catch {
        return null
      }
      let { frames } = scanZstdFrames(head, 1)
      if (frames.length === 0) {
        // Head read did not cover one full frame; take the whole file.
        try {
          const full = await readFile(filePath)
          frames = scanZstdFrames(full, 1).frames
          if (frames.length === 0) return null
          head = full
        } catch {
          return null
        }
      }
      const text = zstdDecompress(head.subarray(frames[0]!.start, frames[0]!.end)).toString('utf-8')
      return text.split('\n').find(l => l.trim()) ?? null
    }
    const content = await readSessionFile(filePath)
    return content?.split('\n').find(l => l.trim()) ?? null
  }

  try {
    const line = await firstLine()
    if (!line) return null
    const event = JSON.parse(line) as DshEvent
    return event.type === 'session' ? event : null
  } catch {
    return null
  }
}

async function discoverSessionsInDir(sessionsDir: string): Promise<SessionSource[]> {
  const sources: SessionSource[] = []

  let projectDirs: string[]
  try {
    projectDirs = await readdir(sessionsDir)
  } catch {
    return sources
  }

  for (const dirName of projectDirs) {
    const dirPath = join(sessionsDir, dirName)
    const dirStat = await stat(dirPath).catch(() => null)
    if (!dirStat?.isDirectory()) continue

    let sessionDirs: string[]
    try {
      sessionDirs = await readdir(dirPath)
    } catch {
      continue
    }

    for (const sessionDir of sessionDirs) {
      const sessionPath = join(dirPath, sessionDir)
      const sessionStat = await stat(sessionPath).catch(() => null)
      if (!sessionStat?.isDirectory()) continue

      // Compressed log first; the uncompressed variant exists when
      // compression=none. Never both for the same session.
      let filePath: string | null = null
      for (const name of ['session.jsonl.zstd', 'session.jsonl']) {
        const candidate = join(sessionPath, name)
        const fileStat = await stat(candidate).catch(() => null)
        if (fileStat?.isFile()) {
          filePath = candidate
          break
        }
      }
      if (!filePath) continue

      const header = await readSessionHeader(filePath)
      if (!header) continue

      const cwd = typeof header.cwd === 'string' && header.cwd.trim() ? header.cwd : dirName
      sources.push({ path: filePath, project: projectFromCwd(cwd, dirName), provider: 'dsh' })
    }
  }

  return sources
}

function parseToolArguments(raw: string | undefined): Record<string, unknown> | null {
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : null
  } catch {
    return null
  }
}

function createParser(source: SessionSource, seenKeys: Set<string>): SessionParser {
  return {
    async *parse(): AsyncGenerator<ParsedProviderCall> {
      const lines = await readEventLines(source.path)
      if (!lines) return

      let sessionId = ''
      let cwd = ''
      let model = 'unknown'
      let currentTurn = 0
      const userMessageByTurn = new Map<number, string>()
      const buckets = new Map<string, StepBucket>()

      for (const line of lines) {
        let event: DshEvent
        try {
          event = JSON.parse(line) as DshEvent
        } catch {
          continue
        }

        if (event.type === 'session') {
          sessionId = event.id ?? sessionId
          cwd = event.cwd ?? cwd
          continue
        }

        if (event.type === 'turn/start') {
          currentTurn = event.data?.turn ?? currentTurn
          continue
        }

        if (event.type === 'request/header') {
          // Emitted at most once per request; steps after the last header
          // inherit its config as their model.
          const headerModel = event.data?.header?.config?.model
          if (typeof headerModel === 'string' && headerModel) model = headerModel
          continue
        }

        if (event.type === 'user/message') {
          const texts = (event.data?.content ?? [])
            .filter(c => c.type === 'text' && typeof c.text === 'string' && c.text)
            .map(c => c.text!)
          if (texts.length > 0) userMessageByTurn.set(currentTurn, texts.join(' '))
          continue
        }

        if (event.type === 'tool/call') {
          const turn = event.data?.turn ?? currentTurn
          const step = event.data?.step ?? 0
          const rawName = event.data?.name
          if (!rawName) continue
          const key = `${turn}:${step}`
          let bucket = buckets.get(key)
          if (!bucket) {
            bucket = { usage: {}, final: false, model, tools: [], skills: [], bashCommands: [] }
            buckets.set(key, bucket)
          }
          bucket.tools.push(mapToolName(rawName))
          const args = parseToolArguments(event.data?.arguments)
          if ((rawName === 'bash' || rawName === 'pwsh') && typeof args?.['command'] === 'string') {
            bucket.bashCommands.push(...extractBashCommands(args['command']))
          }
          if (rawName === 'skill' && typeof args?.['name'] === 'string') {
            bucket.skills.push(args['name'])
          }
          continue
        }

        let usage: DshUsage | undefined
        let isFinal = false
        if (event.type === 'assistant/chunk' && event.data?.chunk?.type === 'usage') {
          usage = event.data.chunk.usage
        } else if (event.type === 'assistant/message' && event.data?.usage) {
          usage = event.data.usage
          isFinal = true
        } else {
          continue
        }
        if (!usage) continue

        const turn = event.data?.turn ?? currentTurn
        const step = event.data?.step ?? 0
        const key = `${turn}:${step}`
        let bucket = buckets.get(key)
        if (!bucket) {
          bucket = { usage: {}, final: false, model, tools: [], skills: [], bashCommands: [] }
          buckets.set(key, bucket)
        }
        // A final report replaces an earlier sample; a late sample never
        // overwrites a final one. The model snapshot follows the winning
        // report (a header can change the model mid-turn between steps).
        if (isFinal || !bucket.final) {
          bucket.usage = usage
          bucket.final = isFinal
          bucket.time = event.time
          bucket.model = model
        }
      }

      const sortedKeys = [...buckets.keys()].sort((a, b) => {
        const [ta, sa] = a.split(':').map(Number)
        const [tb, sb] = b.split(':').map(Number)
        return ta! - tb! || sa! - sb!
      })

      for (const key of sortedKeys) {
        const bucket = buckets.get(key)!
        const input = bucket.usage.inputTokens ?? 0
        const output = bucket.usage.outputTokens ?? 0
        const cacheRead = bucket.usage.cacheReadTokens ?? 0
        const cacheWrite = bucket.usage.cacheWriteTokens ?? 0
        const reasoning = bucket.usage.reasoningTokens ?? 0
        if (input + output + cacheRead + cacheWrite + reasoning === 0) continue

        const dedupKey = `dsh:${sessionId || source.path}:${key}`
        if (seenKeys.has(dedupKey)) continue
        seenKeys.add(dedupKey)

        // DSH bills reasoning tokens at the output rate (same as Gemini).
        const costUSD = calculateCost(bucket.model, input, output + reasoning, cacheWrite, cacheRead, 0)
        const [turn] = key.split(':').map(Number)

        yield {
          provider: 'dsh',
          model: bucket.model,
          inputTokens: input,
          outputTokens: output,
          cacheCreationInputTokens: cacheWrite,
          cacheReadInputTokens: cacheRead,
          cachedInputTokens: cacheRead,
          reasoningTokens: reasoning,
          webSearchRequests: 0,
          costUSD,
          tools: [...new Set(bucket.tools)],
          bashCommands: bucket.bashCommands,
          skills: bucket.skills.length > 0 ? [...new Set(bucket.skills)] : undefined,
          timestamp: typeof bucket.time === 'number' ? new Date(bucket.time).toISOString() : '',
          speed: 'standard',
          deduplicationKey: dedupKey,
          userMessage: userMessageByTurn.get(turn!) ?? '',
          sessionId: sessionId || source.path,
          project: cwd ? projectFromCwd(cwd, source.project) : source.project,
          projectPath: cwd || undefined,
        }
      }
    },
  }
}

export function createDshProvider(dshHomeOverride?: string): Provider {
  const dshHome = getDshHome(dshHomeOverride)
  const sessionsDir = join(dshHome, 'sessions')

  return {
    name: 'dsh',
    displayName: 'DeepSeek Harness',

    modelDisplayName(model: string): string {
      return getShortModelName(model)
    },

    toolDisplayName(rawTool: string): string {
      return mapToolName(rawTool)
    },

    async probeRoots(): Promise<ProbeRoot[]> {
      return [{ path: sessionsDir, label: 'sessions' }]
    },

    async discoverSessions(): Promise<SessionSource[]> {
      return discoverSessionsInDir(sessionsDir)
    },

    createSessionParser(source: SessionSource, seenKeys: Set<string>): SessionParser {
      return createParser(source, seenKeys)
    },
  }
}

export const dsh = createDshProvider()
