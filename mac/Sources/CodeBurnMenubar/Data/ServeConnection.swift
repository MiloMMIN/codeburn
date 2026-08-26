import Darwin
import Foundation

/// A resident `codeburn serve --stdio` child, held so payload fetches skip the
/// per-spawn cost (node boot + a 100MB+ session-cache parse on large corpora,
/// seconds per fetch at the CLI level). Requests are JSON lines `{id, args}`;
/// replies are `{id, ok, output}`. Mirrors the desktop app's client contract:
///
/// - Only `status` payload queries route here; anything else spawns as before.
/// - The first real status request is also the warm-up. It may be written
///   before the child announces READY; the pipe buffers it until serve reads
///   stdin, avoiding a second one-shot process that parses the same cache.
/// - Transport/protocol failures fall back to the spawn path for that call;
///   resource-policy failures remain terminal. Three child deaths disable
///   serve for this app run.
/// - The child's stdin closing (app quit, even SIGKILL) ends the server loop
///   on the CLI side, so no orphan survives the menubar.
actor ServeConnection {
    static let shared = ServeConnection()

    typealias ProcessFactory = ([String], QualityOfService) -> Process
    typealias TimeoutSleep = @Sendable (UInt64) async throws -> Void

    private struct QueuedRequest {
        let token: Int
        let args: [String]
        let continuation: CheckedContinuation<Data, Error>
    }

    private struct ActiveRequest {
        let token: Int
        let id: Int
        let args: [String]
        let child: Process
    }

    private var process: Process?
    private var stdinHandle: FileHandle?
    private var nextId = 1
    private var nextRequestToken = 1
    private var queuedRequests: [QueuedRequest] = []
    private var activeRequest: ActiveRequest?
    private var pending: [Int: CheckedContinuation<Data, Error>] = [:]
    private var requestTimeouts: [Int: Task<Void, Never>] = [:]
    private var timeoutOwners: [Int: Process] = [:]
    private var responseBytes: [Int: Int] = [:]
    private var deaths = 0
    private var buffer = Data()
    private var receivedTerminalResponse = false
    private var outputTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
    private var terminationTasks: [ObjectIdentifier: Task<Void, Never>] = [:]
    private let makeProcess: ProcessFactory
    private let timeoutSleep: TimeoutSleep
    private let terminationGraceSleep: TimeoutSleep
    private let responseLimitBytes: Int

    private static let maxDeaths = 3
    static let maxResponseBytes = 16 * 1024 * 1024
    private static let stdoutReadChunkBytes = 64 * 1024
    private static let terminationGraceNanoseconds: UInt64 = 1_000_000_000
    private static let coldRequestTimeoutNanoseconds: UInt64 = 10 * 60 * 1_000_000_000
    private static let warmRequestTimeoutNanoseconds: UInt64 = 60 * 1_000_000_000

    struct ServeUnavailable: Error {}
    enum FailureReason: Sendable, Equatable {
        case generic
        case outputTooLarge
    }
    struct ServeRequestFailed: Error, Sendable {
        let message: String
        let reason: FailureReason

        init(message: String, reason: FailureReason = .generic) {
            self.message = message
            self.reason = reason
        }
    }

    init(
        makeProcess: @escaping ProcessFactory = CodeburnCLI.makeProcess,
        timeoutSleep: @escaping TimeoutSleep = { nanoseconds in
            try await Task<Never, Never>.sleep(nanoseconds: nanoseconds)
        },
        terminationGraceSleep: @escaping TimeoutSleep = { nanoseconds in
            try await Task<Never, Never>.sleep(nanoseconds: nanoseconds)
        },
        responseLimitBytes: Int = ServeConnection.maxResponseBytes
    ) {
        self.makeProcess = makeProcess
        self.timeoutSleep = timeoutSleep
        self.terminationGraceSleep = terminationGraceSleep
        precondition(responseLimitBytes > 0)
        self.responseLimitBytes = responseLimitBytes
    }

    static func isEligible(_ subcommand: [String]) -> Bool {
        subcommand.first == "status"
    }

    /// Kick the child off (idempotent). Called from app startup and again by
    /// the first request in case the startup task has not run yet.
    func ensureStarted() {
        guard process == nil, deaths < Self.maxDeaths else { return }
        // This single resident serves both background and user-visible status
        // requests. Its cold hydration replaces the old interactive one-shot,
        // so keep the child at the same user-initiated QoS as visible fetches.
        let child = makeProcess(["serve", "--stdio"], .userInitiated)
        let stdinPipe = Pipe()
        let stdinWriter = stdinPipe.fileHandleForWriting
        // Suppress SIGPIPE only for this connection's write end. A process-wide
        // SIG_IGN leaks into unrelated libraries and children; F_SETNOSIGPIPE
        // keeps a closed child stdin on the normal throwable EPIPE path.
        guard Darwin.fcntl(stdinWriter.fileDescriptor, F_SETNOSIGPIPE, 1) == 0 else {
            deaths = Self.maxDeaths
            return
        }
        let stdoutPipe = Pipe()
        let stdoutReader = stdoutPipe.fileHandleForReading
        child.standardInput = stdinPipe
        child.standardOutput = stdoutPipe
        child.standardError = FileHandle.nullDevice
        do {
            try child.run()
        } catch {
            deaths = Self.maxDeaths // spawn path can't produce the binary either better than makeProcess did
            return
        }
        process = child
        stdinHandle = stdinWriter
        let generation = ObjectIdentifier(child)
        // One blocking reader owns this generation's stdout. It never reads a
        // second bounded chunk until the actor has consumed the first, giving
        // the 16 MiB protocol limit real backpressure instead of accumulating
        // an unbounded callback/AsyncStream backlog. EOF is observed only after
        // the pipe's final bytes, so child death cannot overtake a split reply.
        outputTasks[generation] = Task.detached { [weak self] in
            var bytes = [UInt8](repeating: 0, count: Self.stdoutReadChunkBytes)
            while !Task.isCancelled {
                let count = Darwin.read(stdoutReader.fileDescriptor, &bytes, bytes.count)
                if count > 0 {
                    guard let self else { break }
                    await self.consume(Data(bytes[0..<count]), from: child)
                } else if count == -1, errno == EINTR {
                    continue
                } else {
                    break
                }
            }
            await self?.outputStreamEnded(for: child)
            await self?.outputStreamFinished(for: child)
        }
    }

    /// Send the first real payload through the resident child. A request does
    /// not need to wait for the READY frame: stdin is safe to write as soon as
    /// Process.run() succeeds, and serve serializes it after initialization.
    func request(args: [String]) async throws -> Data {
        try Task.checkCancellation()
        ensureStarted()
        guard process != nil else { throw ServeUnavailable() }
        let token = nextRequestToken
        nextRequestToken += 1
        let response = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queuedRequests.append(QueuedRequest(
                    token: token,
                    args: args,
                    continuation: continuation
                ))
                startNextRequestIfPossible()
            }
        } onCancel: {
            Task { await self.cancelRequest(token: token) }
        }
        try Task.checkCancellation()
        return response
    }

    func shutdown() {
        deaths = Self.maxDeaths
        process?.terminate()
        for task in terminationTasks.values { task.cancel() }
        terminationTasks.removeAll()
        cancelAllTimeouts()
        failAllRequests()
        process = nil
        stdinHandle = nil
        buffer = Data()
        receivedTerminalResponse = false
    }

    // MARK: - internals

    private func startNextRequestIfPossible() {
        guard activeRequest == nil, !queuedRequests.isEmpty else { return }
        ensureStarted()
        guard let stdinHandle, let child = process else {
            failQueuedRequests(error: ServeUnavailable())
            return
        }
        // A Process can report not-running just before its termination callback
        // reaches the ordered event stream. Keep the request queued for that
        // event instead of writing to a generation which is already exiting.
        guard child.isRunning else { return }

        let request = queuedRequests.removeFirst()
        let id = nextId
        nextId += 1
        let line: Data
        do {
            line = try JSONSerialization.data(withJSONObject: ["id": id, "args": request.args])
        } catch {
            request.continuation.resume(throwing: error)
            startNextRequestIfPossible()
            return
        }

        // The previous response can resume its caller just before EOF reaches
        // this actor. Avoid admitting a successor to an already-reaped child;
        // the reader's ordered EOF path will start it on a replacement.
        guard child.isRunning else {
            queuedRequests.insert(request, at: 0)
            outputStreamEnded(for: child)
            return
        }

        // Select and arm the timeout only when this request becomes the sole
        // protocol request in flight. A queued request must not spend its own
        // budget while its predecessor is still hydrating or draining.
        let timeoutNanoseconds = receivedTerminalResponse
            ? Self.warmRequestTimeoutNanoseconds
            : Self.coldRequestTimeoutNanoseconds
        activeRequest = ActiveRequest(
            token: request.token,
            id: id,
            args: request.args,
            child: child
        )
        pending[id] = request.continuation
        responseBytes[id] = 0
        do {
            try stdinHandle.write(contentsOf: line + Data("\n".utf8))
            armTimeout(id: id, child: child, nanoseconds: timeoutNanoseconds)
        } catch {
            // The previous terminal frame can resume its caller just before
            // EOF detaches that generation. Preserve this never-admitted
            // request and retry it on the replacement instead of surfacing a
            // transient EPIPE to the UI.
            pending.removeValue(forKey: id)
            responseBytes.removeValue(forKey: id)
            activeRequest = nil
            queuedRequests.insert(request, at: 0)
            outputStreamEnded(for: child)
        }
    }

    private func cancelRequest(token: Int) {
        if let index = queuedRequests.firstIndex(where: { $0.token == token }) {
            let request = queuedRequests.remove(at: index)
            request.continuation.resume(throwing: CancellationError())
            return
        }
        guard let activeRequest, activeRequest.token == token,
              let continuation = pending.removeValue(forKey: activeRequest.id) else { return }
        continuation.resume(throwing: CancellationError())
        // Caller cancellation abandons only this response. The serialized serve
        // child may still be doing the expensive first hydration, and killing it
        // here lets tab switches and UI watchdogs restart that work indefinitely.
        // Its independent request timeout remains armed: a command that never
        // returns is still reaped, so it cannot wedge every later serialized call.
    }

    private func armTimeout(id: Int, child: Process, nanoseconds: UInt64) {
        let sleep = timeoutSleep
        timeoutOwners[id] = child
        requestTimeouts[id] = Task.detached { [weak self] in
            do {
                try await sleep(nanoseconds)
            } catch {
                return
            }
            await self?.requestTimedOut(id: id)
        }
    }

    private func requestTimedOut(id: Int) {
        guard let child = timeoutOwners.removeValue(forKey: id) else { return }
        requestTimeouts.removeValue(forKey: id)
        responseBytes.removeValue(forKey: id)
        if let continuation = pending.removeValue(forKey: id) {
            continuation.resume(throwing: ServeRequestFailed(message: "serve timeout"))
        }
        // The waiter may already have been abandoned by caller cancellation.
        // Timeout ownership is deliberately independent of that continuation:
        // kill only the exact generation that received the timed-out request.
        guard process === child else {
            if activeRequest?.id == id { activeRequest = nil }
            startNextRequestIfPossible()
            return
        }
        // Retire the timed-out generation synchronously. Its stdout may never
        // reach EOF (for example, a stuck child can ignore SIGTERM or a
        // descendant can retain the pipe), so waiting for the reader would also
        // spend every queued caller's timeout before it can even be admitted.
        process = nil
        stdinHandle = nil
        buffer = Data()
        receivedTerminalResponse = false
        deaths += 1
        if activeRequest?.id == id { activeRequest = nil }
        cancelTimeouts(ownedBy: child)
        terminateTimedOutChild(child)
        // The waiter was removed above and cannot be requeued by stale EOF.
        // A queued read starts on a replacement immediately, subject to the
        // ordinary three-death budget.
        startNextRequestIfPossible()
    }

    private func cancelTimeout(id: Int) {
        timeoutOwners.removeValue(forKey: id)
        requestTimeouts.removeValue(forKey: id)?.cancel()
        responseBytes.removeValue(forKey: id)
    }

    private func cancelTimeouts(ownedBy child: Process) {
        let ids = timeoutOwners.compactMap { id, owner in owner === child ? id : nil }
        for id in ids { cancelTimeout(id: id) }
    }

    private func cancelAllTimeouts() {
        for task in requestTimeouts.values { task.cancel() }
        requestTimeouts.removeAll()
        timeoutOwners.removeAll()
        responseBytes.removeAll()
    }

    private func outputStreamFinished(for child: Process) {
        outputTasks.removeValue(forKey: ObjectIdentifier(child))
        if !child.isRunning {
            terminationTasks.removeValue(forKey: ObjectIdentifier(child))?.cancel()
        }
    }

    private func terminateTimedOutChild(_ child: Process) {
        guard child.isRunning else { return }
        child.terminate()
        let generation = ObjectIdentifier(child)
        let sleep = terminationGraceSleep
        terminationTasks[generation] = Task.detached { [weak self] in
            do {
                try await sleep(Self.terminationGraceNanoseconds)
            } catch {
                // Cancellation means the owner stopped waiting: either shutdown
                // (which must not orphan a SIGTERM-ignoring generation) or the
                // child already died and the stream finished. Escalate either
                // way; the isRunning guard makes the dead-child case a no-op.
                await self?.forceKillAfterGrace(child)
                return
            }
            await self?.forceKillAfterGrace(child)
        }
    }

    private func forceKillAfterGrace(_ child: Process) {
        terminationTasks.removeValue(forKey: ObjectIdentifier(child))
        guard child.isRunning else { return }
        _ = Darwin.kill(child.processIdentifier, SIGKILL)
    }

    private func outputStreamEnded(for child: Process) {
        guard process === child else { return }
        // EOF/read failure is a transport death even if the process has not
        // reaped yet. Terminate that exact generation so a child which closed
        // stdout cannot survive after the actor starts its replacement.
        if child.isRunning { child.terminate() }
        childDied(child)
    }

    // Internal so the generation guard can be exercised deterministically by
    // tests without relying on Foundation callback scheduling at process exit.
    func consume(_ data: Data, from child: Process) {
        // A readability callback can already have queued its actor Task when the
        // old process exits. If a replacement starts first, those late bytes must
        // not repopulate the shared line buffer or mark the new child as warm.
        guard process === child else { return }
        var remaining = data[data.startIndex..<data.endIndex]
        while !remaining.isEmpty {
            if let newline = remaining.firstIndex(of: UInt8(ascii: "\n")) {
                let fragment = remaining[remaining.startIndex..<newline]
                guard fragment.count <= responseLimitBytes - buffer.count else {
                    outputOverflowed(child)
                    return
                }
                buffer.append(contentsOf: fragment)
                let lineData = buffer
                buffer = Data()
                consumeLine(lineData, from: child)
                guard process === child else { return }
                remaining = remaining[remaining.index(after: newline)..<remaining.endIndex]
            } else {
                guard remaining.count <= responseLimitBytes - buffer.count else {
                    outputOverflowed(child)
                    return
                }
                buffer.append(contentsOf: remaining)
                return
            }
        }
    }

    private func consumeLine(_ lineData: Data, from child: Process) {
        guard !lineData.isEmpty,
              let object = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else { return }
            if object["ready"] as? Bool == true {
                return
            }
            guard let id = object["id"] as? Int,
                  responseBytes[id] != nil else { return }
            // Desktop asks serve to stream cold-scan stderr as progress frames.
            // Menubar has no progress UI, but must leave the request pending
            // until the terminal response arrives if such a frame is emitted.
            if let progress = object["progress"] as? String {
                guard accountResponseBytes(Data(progress.utf8).count, id: id, child: child) else { return }
                return
            }
            let succeeded = object["ok"] as? Bool == true
            let payload = succeeded
                ? (object["output"] as? String)
                : (object["error"] as? String)
            if let payload {
                guard accountResponseBytes(Data(payload.utf8).count, id: id, child: child) else { return }
            }
            // A refused/failed command can finish before any cache hydration.
            // Only a successful terminal proves the resident is warm. Keep
            // this before the waiter lookup so a successful orphan response
            // still records the child as warm without resuming anything.
            if succeeded { receivedTerminalResponse = true }
            cancelTimeout(id: id)
            let continuation = pending.removeValue(forKey: id)
            if activeRequest?.id == id { activeRequest = nil }
            if let continuation {
                if succeeded, let output = object["output"] as? String {
                    continuation.resume(returning: Data(output.utf8))
                } else {
                    let message = object["error"] as? String ?? "serve request failed"
                    continuation.resume(throwing: ServeRequestFailed(message: message))
                }
            }
            // This also advances after an orphan terminal response whose caller
            // was cancelled: cancellation removes only the waiter, not the
            // active protocol lifecycle.
            startNextRequestIfPossible()
    }

    private func accountResponseBytes(_ count: Int, id: Int, child: Process) -> Bool {
        guard let current = responseBytes[id],
              count <= responseLimitBytes - current else {
            outputOverflowed(child)
            return false
        }
        responseBytes[id] = current + count
        return true
    }

    private func outputOverflowed(_ child: Process) {
        guard process === child else { return }
        // Detach this exact generation before terminating it. Its eventual exit
        // and any already-scheduled stdout callbacks are then stale and cannot
        // consume a second death or corrupt a replacement generation.
        process = nil
        stdinHandle = nil
        buffer = Data()
        receivedTerminalResponse = false
        deaths += 1
        cancelTimeouts(ownedBy: child)
        failAllRequests(error: ServeRequestFailed(
            message: "serve output exceeded \(responseLimitBytes) bytes",
            reason: .outputTooLarge
        ))
        if child.isRunning { child.terminate() }
    }

    private func childDied(_ child: Process) {
        guard process === child else { return }
        process = nil
        stdinHandle = nil
        buffer.removeAll()
        receivedTerminalResponse = false
        deaths += 1
        cancelTimeouts(ownedBy: child)
        if let activeRequest, activeRequest.child === child {
            if let continuation = pending.removeValue(forKey: activeRequest.id) {
                // Only read-only status requests enter this connection. If a
                // generation exits after admission but before its terminal
                // reply, retain the waiter and retry on the replacement rather
                // than racing it into a one-shot fallback. A timed-out or
                // cancelled waiter is already absent and is never retried.
                queuedRequests.insert(QueuedRequest(
                    token: activeRequest.token,
                    args: activeRequest.args,
                    continuation: continuation
                ), at: 0)
            }
            self.activeRequest = nil
        }
        // Requests which were never written survive an ordinary child crash.
        // They begin on a replacement only after this ordered death event.
        startNextRequestIfPossible()
    }

    private func failAllRequests(
        error: Error = ServeRequestFailed(message: "serve exited")
    ) {
        for (_, continuation) in pending {
            continuation.resume(throwing: error)
        }
        pending.removeAll()
        activeRequest = nil
        failQueuedRequests(error: error)
    }

    private func failQueuedRequests(error: Error) {
        let requests = queuedRequests
        queuedRequests.removeAll()
        for request in requests {
            request.continuation.resume(throwing: error)
        }
    }
}
