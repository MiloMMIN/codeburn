import { readdir, stat } from "fs/promises";
import { statSync } from "fs";
import { basename, join } from "path";
import { homedir } from "os";

import { calculateCost, getShortModelName } from "../models.js";
import { blobToText, isSqliteAvailable, openDatabase } from "../sqlite.js";
import { readConfig } from "../config.js";
import type {
  ProbeRoot,
  Provider,
  SessionParser,
  SessionSource,
  ParsedProviderCall,
} from "./types.js";
import type { DateRange } from "../types.js";
import { readSessionFile } from "../fs-utils.js";
import { isPositiveNumber, safeNumber } from "../parser.js";

type AgentTrajectory<StepType extends Step = Step, AgentExtra = unknown> = {
  schema_version: string;
  session_id?: string;
  agent: Agent<AgentExtra>;
  steps: StepType[];
  final_metrics?: FinalMetrics;
};

type FinalMetrics = {
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  total_cached_tokens?: number;
  total_steps?: number;
};

type DevinAgentExtra = {
  backend?: string;
  permission_mode?: string;
};

type Agent<Extra = unknown> = {
  name: string;
  version: string;
  model_name?: string;
  tool_definitions?: unknown;
  extra?: Extra;
};

type ToolCall = {
  tool_call_id: string;
  function_name: string;
  arguments: unknown;
};

type DevinMetadata = {
  created_at?: string;
  committed_acu_cost?: number;
  generation_model?: string;
  is_user_input?: boolean;
  num_tokens?: number;
  request_id?: string;
  finish_reason?: string;
  metrics?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_creation_tokens?: number;
    cache_read_tokens?: number;
    tokens_per_sec?: number;
    total_time_ms?: number;
    ttft_ms?: number;
    tpot_ms?: number;
  };
};

type ContentPart = ContentPartText | ContentPartImage;

type ContentPartText = {
  type: "text";
  text: string;
};

type ContentPartImage = {
  type: "image";
  source: ImageSource;
};

function isTextContentPart(
  contentPart: ContentPart,
): contentPart is ContentPartText {
  return contentPart.type === "text";
}

type ImageSource = {
  media_type: string;
  path: string;
};

type Step<StepExtra = unknown, MetricsExtra = unknown> = {
  step_id: number;
  timestamp?: string;
  source: string;
  model_name?: string;
  message: string | Array<ContentPart>;
  tool_calls?: Array<ToolCall>;
  extra?: StepExtra;
  observation?: Observation;
  metrics?: Metrics<MetricsExtra>;
};

type DevinTelemetry = {
  source?: string;
  operation?: string;
};

type DevinStepExtra = {
  committed_acu_cost?: number;
  generation_model?: string;
  telemetry?: DevinTelemetry;
};

type Observation = {
  results: Array<ObservationResult>;
};

type ObservationResult = {
  source_call_id?: string;
  content?: string | Array<ContentPart>;
};

type Metrics<Extra = unknown> = {
  prompt_tokens?: number;
  completion_tokens?: number;
  cached_tokens?: number;
  extra?: Extra;
};

type DevinMetricsExtra = {
  cache_creation_input_tokens?: number;
};

type DevinStep = Step<DevinStepExtra, DevinMetricsExtra> & {
  metadata?: DevinMetadata;
};

type DevinAgentTrajectory = AgentTrajectory<DevinStep, DevinAgentExtra>;

type DevinSessionMetadata = {
  id: string;
  workingDirectory: string;
  model: string;
  title?: string;
  createdAt: string;
  lastActivityAt: string;
  hidden: boolean;
};

type DevinUsage = {
  committedAcuCost: number;
  inputTokens: number;
  outputTokens: number;
  cacheCreationInputTokens: number;
  cacheReadInputTokens: number;
};

// Devin Desktop keeps its CLI data under %APPDATA% on Windows; the Linux-style
// XDG path is the fallback there and the default everywhere else.
const DEFAULT_DEVIN_CLI_DIRS =
  process.platform === "win32"
    ? [
        join(homedir(), "AppData", "Roaming", "devin", "cli"),
        join(homedir(), ".local", "share", "devin", "cli"),
      ]
    : [join(homedir(), ".local", "share", "devin", "cli")];

const DEFAULT_MODEL_NAME = "devin";
const DEVIN_PROVIDER_NAME = "devin";
const DEVIN_PROVIDER_DISPLAY_NAME = "Devin";
const DEVIN_TRANSCRIPTS_SUBDIR = "transcripts";
const DEVIN_SESSIONS_DB = "sessions.db";
const DEVIN_EFFORT_TIERS = new Set(["xhigh", "high", "medium", "low"]);

function parseTranscript(raw: string): DevinAgentTrajectory | null {
  try {
    return JSON.parse(raw) as DevinAgentTrajectory;
  } catch {
    return null;
  }
}

function parseNumericTimestamp(value: number): string {
  const millis = value < 10_000_000_000 ? value * 1000 : value;
  return new Date(millis).toISOString();
}

function getCommittedAcuCost(step: DevinStep): number {
  const acuCost = [
    step.metadata?.committed_acu_cost,
    step.extra?.committed_acu_cost,
  ].filter((cost) => isPositiveNumber(cost));

  return acuCost.shift() || 0;
}

function hasAnyTokenField(
  metrics: Metrics<DevinMetricsExtra> | null | undefined,
): boolean {
  if (!metrics) return false;
  return [
    metrics.prompt_tokens,
    metrics.completion_tokens,
    metrics.cached_tokens,
    metrics.extra?.cache_creation_input_tokens,
  ].some((value) => value != null);
}

function getMetricsFromStep(
  step: DevinStep,
): Metrics<DevinMetricsExtra> | null {
  // Prefer step.metrics (standard ATIF v1.7) only when it actually carries
  // token fields; a present-but-empty metrics object must not shadow the
  // legacy metadata.metrics location.
  if (hasAnyTokenField(step.metrics)) {
    return step.metrics ?? null;
  }

  if (step.metadata) {
    return getDevinMetricsFromMetadata(step.metadata);
  }

  return step.metrics ?? null;
}

function getDevinMetricsFromMetadata(
  metadata: DevinMetadata,
): Metrics<DevinMetricsExtra> {
  return {
    prompt_tokens: metadata.metrics?.input_tokens,
    completion_tokens: metadata.metrics?.output_tokens,
    cached_tokens: metadata.metrics?.cache_read_tokens,
    extra: {
      cache_creation_input_tokens: metadata.metrics?.cache_creation_tokens,
    },
  };
}

function getUsage(step: DevinStep): DevinUsage | null {
  const committedAcuCost = getCommittedAcuCost(step);
  const metrics = getMetricsFromStep(step);

  const hasAnyUsage = [
    committedAcuCost,
    metrics?.prompt_tokens,
    metrics?.completion_tokens,
    metrics?.extra?.cache_creation_input_tokens,
    metrics?.cached_tokens,
  ].some((x) => isPositiveNumber(x));

  if (!hasAnyUsage) return null;

  return {
    committedAcuCost,
    inputTokens: safeNumber(metrics?.prompt_tokens),
    outputTokens: safeNumber(metrics?.completion_tokens),
    cacheCreationInputTokens: safeNumber(
      metrics?.extra?.cache_creation_input_tokens,
    ),
    cacheReadInputTokens: safeNumber(metrics?.cached_tokens),
  };
}

function getSessionId(
  source: SessionSource,
  transcript: DevinAgentTrajectory,
): string {
  const fromTranscript = transcript.session_id?.trim();
  return fromTranscript || basename(source.path, ".json");
}

function projectNameFromPath(path: string): string {
  const normalized = path.trim().replace(/[/\\]+$/, "");
  return normalized.split(/[/\\]/).filter(Boolean).pop() ?? path;
}

function getProjectName(
  source: SessionSource,
  session: DevinSessionMetadata | null,
): string {
  if (session?.workingDirectory)
    return projectNameFromPath(session.workingDirectory);
  if (session?.title) return session.title;
  return source.project;
}

function getProjectPath(
  session: DevinSessionMetadata | null,
): string | undefined {
  return session?.workingDirectory;
}

function getTimestamp(
  step: DevinStep,
  session: DevinSessionMetadata | null,
): string | undefined {
  return [
    step.metadata?.created_at,
    session?.lastActivityAt,
    session?.createdAt,
  ]
    .filter(Boolean)
    .shift();
}

function firstPresentString(...values: Array<string | undefined>): string | undefined {
  for (const value of values) {
    const trimmed = value?.trim();
    if (trimmed) return trimmed;
  }
  return undefined;
}

function getFriendlyGptName(model: string): string {
  const shortName = getShortModelName(model);
  const match = model.match(/^gpt-(\d+(?:\.\d+)*)(?:-(.+))?$/);
  if (!match) return shortName;

  const suffixParts = match[2]?.split("-").filter(Boolean) ?? [];
  // A purely numeric suffix token means this is a dated snapshot id such as
  // gpt-4-1106-preview, not a clean version+word id. Fabricating a friendly
  // name here would mislabel the date as text (e.g. "GPT-4 1106 Preview"), so
  // defer to getShortModelName, which passes unknown snapshots through raw.
  if (suffixParts.some((part) => /^\d+$/.test(part))) return shortName;

  const suffix = suffixParts
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

  const reconstructed = `GPT-${match[1]}${suffix ? ` ${suffix}` : ""}`;
  if (shortName !== model && (!suffix || shortName !== `GPT-${match[1]}`)) {
    return shortName;
  }

  return reconstructed;
}

function getDevinDisplayModelName(
  generationModel: string | undefined,
  modelName: string,
): string {
  if (!generationModel || /^MODEL_/.test(generationModel)) {
    return getShortModelName(modelName);
  }

  if (generationModel.startsWith("gpt-")) {
    // Devin minor versions are always a single digit (gpt-5-3-codex). Restrict
    // the dash-to-dot rewrite to a single-digit minor at a token boundary so a
    // dated snapshot like gpt-4-1106-preview is not misread as version 4.1106.
    const normalized = generationModel.replace(/^gpt-(\d+)-(\d)(?=-|$)/, "gpt-$1.$2");
    const effortMatch = normalized.match(/-([^-]+)$/);
    const effort = effortMatch && DEVIN_EFFORT_TIERS.has(effortMatch[1]!)
      ? effortMatch[1]
      : undefined;
    const base = effort ? normalized.slice(0, -(effort.length + 1)) : normalized;
    const friendlyBase = getFriendlyGptName(base);
    return effort ? `${friendlyBase} (${effort})` : friendlyBase;
  }

  return getShortModelName(generationModel);
}

function getModelName(
  transcript: DevinAgentTrajectory,
  step: DevinStep,
  session: DevinSessionMetadata | null,
): string {
  const generationModel = firstPresentString(
    step.metadata?.generation_model,
    step.extra?.generation_model,
  );
  const modelName = firstPresentString(
    step.model_name,
    transcript.agent?.model_name,
    session?.model,
  ) ?? DEFAULT_MODEL_NAME;

  return getDevinDisplayModelName(generationModel, modelName);
}

function getToolNames(step: DevinStep): string[] {
  return (step.tool_calls ?? []).map((call) => call.function_name);
}

function normalizeContentPartMessage(contentPart: ContentPart) {
  if (isTextContentPart(contentPart)) {
    return contentPart.text;
  } else {
    return contentPart.source.path;
  }
}

function normalizeStepMessage(message: string | Array<ContentPart>): string {
  if (Array.isArray(message)) {
    return message.map((x) => normalizeContentPartMessage(x).trim()).join(" ");
  }
  return message.trim();
}

function getFirstUserMessageBeforeStep(
  steps: DevinStep[],
  index: number,
): string | null {
  for (let i = index - 1; i >= 0; i--) {
    const step = steps[i];
    if (!step?.metadata?.is_user_input) continue;
    const message = step.message
      ? normalizeStepMessage(step.message)
      : undefined;
    if (message) return message;
  }
  return null;
}

function loadSessionMetadata(
  dbPath: string,
): Map<string, DevinSessionMetadata> {
  const sessions = new Map<string, DevinSessionMetadata>();
  let db: ReturnType<typeof openDatabase> | null = null;
  try {
    db = openDatabase(dbPath);
    const rows = db.query<{
      id: string;
      working_directory: string;
      model: string;
      title: string | null;
      created_at: number;
      last_activity_at: number;
      hidden: number;
    }>(
      `SELECT id, working_directory, model, title, created_at, last_activity_at, hidden
       FROM sessions`,
    );
    for (const row of rows) {
      if (!row.id) continue;
      sessions.set(row.id, {
        id: row.id,
        workingDirectory: row.working_directory,
        model: row.model,
        title: row.title ?? undefined,
        createdAt: parseNumericTimestamp(row.created_at),
        lastActivityAt: parseNumericTimestamp(row.last_activity_at),
        hidden: !!row.hidden,
      });
    }
  } catch {
    return sessions;
  } finally {
    db?.close();
  }
  return sessions;
}

async function getCostFactor(): Promise<number | null> {
  const configRate = (await readConfig()).devin?.acuUsdRate;
  return isPositiveNumber(configRate) ? configRate : null;
}

class DevinSessionParser implements SessionParser {
  constructor(
    private source: SessionSource,
    private seenKeys: Set<string>,
    private sessionMetadata: Map<string, DevinSessionMetadata>,
  ) {}

  async *parse(): AsyncGenerator<ParsedProviderCall> {
    const raw = await readSessionFile(this.source.path);
    if (!raw) return;

    const transcript = parseTranscript(raw);
    if (!transcript?.steps) return;

    const sessionId = getSessionId(this.source, transcript);
    const session = this.sessionMetadata.get(sessionId) ?? null;
    if (session?.hidden) return;

    const project = getProjectName(this.source, session);
    const projectPath = getProjectPath(session);
    const costFactor = await getCostFactor();
    if (costFactor === null) return;

    for (let index = 0; index < transcript.steps.length; index++) {
      const step = transcript.steps[index];
      if (step.metadata?.is_user_input) continue;

      const usage = getUsage(step);
      if (!usage) continue;

      const timestamp = getTimestamp(step, session) ?? "";

      const deduplicationKey = `devin:${sessionId}:${step.step_id}`;

      if (this.seenKeys.has(deduplicationKey)) continue;
      this.seenKeys.add(deduplicationKey);

      const model = getModelName(transcript, step, session);
      const tools = getToolNames(step);
      const userMessage =
        getFirstUserMessageBeforeStep(transcript.steps, index) ?? "";

      yield {
        provider: DEVIN_PROVIDER_NAME,
        model,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        cacheCreationInputTokens: usage.cacheCreationInputTokens,
        cacheReadInputTokens: usage.cacheReadInputTokens,
        cachedInputTokens: usage.cacheReadInputTokens,
        reasoningTokens: 0,
        webSearchRequests: 0,
        costUSD: usage.committedAcuCost * costFactor,
        tools,
        bashCommands: [],
        timestamp,
        speed: "standard",
        deduplicationKey,
        userMessage,
        sessionId,
        project,
        projectPath,
      };
    }
  }
}

// ── sessions.db message tree ─────────────────────────────────────────────
// Current CLI builds keep the whole conversation tree inside sessions.db
// (message_nodes.chat_message); no transcripts/ directory is written. Each
// assistant node carries metadata.metrics (input/output/cache tokens),
// metadata.generation_model (canonical model id), and request_id.
// The node tree shares messages across branches after a fork, so the same
// API call appears on multiple rows — dedupe on request_id, not row_id.

type DevinMessageNodeRow = {
  session_id: string | null;
  node_id: number;
  created_at: number | null;
  chat_message: string | Uint8Array | null;
};

type DevinChatMessage = {
  message_id?: string;
  role?: string;
  tool_calls?: Array<{ name?: string; function_name?: string }>;
  metadata?: DevinMetadata;
};

class DevinDbSessionParser implements SessionParser {
  constructor(
    private source: SessionSource,
    private seenKeys: Set<string>,
    private sessionMetadata: Map<string, DevinSessionMetadata>,
    private transcriptSessionIds: Set<string>,
    private dateRange?: DateRange,
  ) {}

  async *parse(): AsyncGenerator<ParsedProviderCall> {
    if (!isSqliteAvailable()) return;
    const costFactor = await getCostFactor();
    let db: ReturnType<typeof openDatabase> | null = null;
    try {
      db = openDatabase(this.source.path);
      const rows = db.query<DevinMessageNodeRow>(
        `SELECT session_id, node_id, created_at, CAST(chat_message AS BLOB) AS chat_message
         FROM message_nodes
         WHERE json_extract(chat_message, '$.role') = 'assistant'
           AND json_extract(chat_message, '$.metadata.metrics') IS NOT NULL` +
          (this.dateRange ? " AND created_at BETWEEN ? AND ?" : ""),
        this.dateRange
          ? [
              Math.floor(this.dateRange.start.getTime() / 1000),
              Math.ceil(this.dateRange.end.getTime() / 1000),
            ]
          : [],
      );
      for (const row of rows) {
        const sessionId = row.session_id ?? "";
        if (!sessionId || this.transcriptSessionIds.has(sessionId)) continue;
        const session = this.sessionMetadata.get(sessionId) ?? null;
        if (session?.hidden) continue;

        let msg: DevinChatMessage;
        try {
          msg = JSON.parse(blobToText(row.chat_message)) as DevinChatMessage;
        } catch {
          continue;
        }
        const metrics = msg.metadata?.metrics;
        if (!metrics) continue;
        const inputTokens = safeNumber(metrics.input_tokens);
        const outputTokens = safeNumber(metrics.output_tokens);
        const cacheReadTokens = safeNumber(metrics.cache_read_tokens);
        const cacheCreationTokens = safeNumber(metrics.cache_creation_tokens);
        if (!inputTokens && !outputTokens && !cacheReadTokens && !cacheCreationTokens) {
          continue;
        }

        const requestId =
          firstPresentString(msg.metadata?.request_id, msg.message_id) ??
          `node-${row.node_id}`;
        const deduplicationKey = `devin:db:${sessionId}:${requestId}`;
        if (this.seenKeys.has(deduplicationKey)) continue;
        this.seenKeys.add(deduplicationKey);

        const model =
          firstPresentString(
            msg.metadata?.generation_model,
            session?.model,
          ) ?? DEFAULT_MODEL_NAME;
        const committedAcuCost = isPositiveNumber(
          msg.metadata?.committed_acu_cost,
        )
          ? (msg.metadata?.committed_acu_cost ?? 0)
          : 0;
        const billedByAcu = committedAcuCost > 0 && costFactor !== null;
        const timestamp =
          typeof row.created_at === "number"
            ? parseNumericTimestamp(row.created_at)
            : (session?.lastActivityAt ?? "");

        yield {
          provider: DEVIN_PROVIDER_NAME,
          model,
          inputTokens,
          outputTokens,
          cacheCreationInputTokens: cacheCreationTokens,
          cacheReadInputTokens: cacheReadTokens,
          cachedInputTokens: cacheReadTokens,
          reasoningTokens: 0,
          webSearchRequests: 0,
          costUSD: billedByAcu
            ? committedAcuCost * costFactor
            : calculateCost(
                model,
                inputTokens,
                outputTokens,
                cacheCreationTokens,
                cacheReadTokens,
                0,
              ),
          costIsEstimated: billedByAcu ? undefined : true,
          tools: (msg.tool_calls ?? [])
            .map((call) => call.name ?? call.function_name ?? "")
            .filter(Boolean),
          bashCommands: [],
          timestamp,
          speed: "standard",
          deduplicationKey,
          userMessage: "",
          sessionId,
          project: getProjectName(this.source, session),
          projectPath: getProjectPath(session),
        };
      }
    } catch {
      // Missing table on an older schema, or a locked/corrupt db — nothing
      // to report.
    } finally {
      db?.close();
    }
  }
}

function resolveDevinCliDir(override?: string): string {
  if (override && override.trim()) return override;
  for (const dir of DEFAULT_DEVIN_CLI_DIRS) {
    try {
      if (statSync(dir).isDirectory()) return dir;
    } catch {
      // keep looking
    }
  }
  return DEFAULT_DEVIN_CLI_DIRS[0];
}

async function isFile(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

function hasMessageNodes(dbPath: string): boolean {
  if (!isSqliteAvailable()) return false;
  let db: ReturnType<typeof openDatabase> | null = null;
  try {
    db = openDatabase(dbPath);
    db.query("SELECT 1 FROM message_nodes LIMIT 1");
    return true;
  } catch {
    return false;
  } finally {
    db?.close();
  }
}

function getDevinDiscoveryRoots(cliDir: string): {
  transcriptsDir: string;
  sessionsDbPath: string;
} {
  return {
    transcriptsDir: join(cliDir, DEVIN_TRANSCRIPTS_SUBDIR),
    sessionsDbPath: join(cliDir, DEVIN_SESSIONS_DB),
  };
}

export function createDevinProvider(cliDir?: string): Provider {
  const resolvedCliDir = resolveDevinCliDir(cliDir);
  const { transcriptsDir, sessionsDbPath } =
    getDevinDiscoveryRoots(resolvedCliDir);
  let sessionMetadata: Map<string, DevinSessionMetadata> | null = null;
  let transcriptSessionIds = new Set<string>();

  const getSessionMetadata = () => {
    if (!sessionMetadata) sessionMetadata = loadSessionMetadata(sessionsDbPath);
    return sessionMetadata;
  };

  return {
    name: DEVIN_PROVIDER_NAME,
    displayName: DEVIN_PROVIDER_DISPLAY_NAME,
    // The CLI owns sessions.db and can prune old sessions; cached calls keep
    // contributing after their rows disappear.
    durableSources: true,

    modelDisplayName(model: string): string {
      return model;
    },

    toolDisplayName(rawTool: string): string {
      return rawTool;
    },

    async probeRoots(): Promise<ProbeRoot[]> {
      return [
        { path: transcriptsDir, label: "transcripts" },
        { path: sessionsDbPath, label: "sessions.db" },
      ];
    },

    async discoverSessions(): Promise<SessionSource[]> {
      const metadata = getSessionMetadata();
      const sources: SessionSource[] = [];
      const covered = new Set<string>();

      // Transcript files price exclusively through committed_acu_cost, so they
      // are only usable once devin.acuUsdRate is configured.
      if ((await getCostFactor()) !== null) {
        const entries = await readdir(transcriptsDir).catch(() => []);

        for (const entry of entries) {
          if (!entry.endsWith(".json")) continue;

          const filePath = join(transcriptsDir, entry);
          const pathStats = await stat(filePath).catch(() => null);

          if (!pathStats?.isFile()) continue;

          const session = metadata.get(basename(filePath, ".json")) ?? null;
          if (session?.hidden) continue;

          const tmpSource: SessionSource = {
            path: filePath,
            project: DEVIN_PROVIDER_NAME,
            provider: DEVIN_PROVIDER_NAME,
          };

          const project = getProjectName(tmpSource, session);

          sources.push({
            path: filePath,
            project,
            provider: DEVIN_PROVIDER_NAME,
          });
          covered.add(basename(filePath, ".json"));
        }
      }
      transcriptSessionIds = covered;

      // Current CLI builds keep the whole message tree in sessions.db — usable
      // even without acuUsdRate, because per-call token metrics support a
      // calculated (estimated) price. Sessions that also have a transcript
      // file are skipped inside the DB parser so nothing counts twice.
      // A db without message_nodes only carries session metadata for the
      // transcript path; probing it keeps discovery cheap for that layout.
      if (await isFile(sessionsDbPath) && hasMessageNodes(sessionsDbPath)) {
        sources.push({
          path: sessionsDbPath,
          project: DEVIN_PROVIDER_NAME,
          provider: DEVIN_PROVIDER_NAME,
          sourceId: "sessions-db",
        });
      }

      return sources;
    },

    createSessionParser(
      source: SessionSource,
      seenKeys: Set<string>,
      dateRange?: DateRange,
    ): SessionParser {
      if (source.sourceId === "sessions-db") {
        return new DevinDbSessionParser(
          source,
          seenKeys,
          getSessionMetadata(),
          transcriptSessionIds,
          dateRange,
        );
      }
      return new DevinSessionParser(source, seenKeys, getSessionMetadata());
    },
  };
}

export const devin = createDevinProvider();
