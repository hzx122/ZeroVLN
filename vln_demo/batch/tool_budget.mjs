// Harness-only execution boundary: no prompt, schema or model parameter changes.
export function installToolBudget(session, maximum) {
  if (!Number.isInteger(maximum) || maximum < 1) throw new Error('max tool calls must be positive');
  const admitted = new Set();
  const completed = new Set();
  let reached = false;
  const before = session.agent.beforeToolCall;
  session.agent.beforeToolCall = async (context, signal) => {
    if (!admitted.has(context.toolCall.id)) return { block: true, reason: 'Tool call budget exhausted' };
    return before?.(context, signal);
  };
  session.subscribe(event => {
    if (event.type === 'tool_execution_start' && admitted.size < maximum) admitted.add(event.toolCallId);
    if (event.type === 'tool_execution_end' && admitted.has(event.toolCallId)) {
      completed.add(event.toolCallId);
      if (completed.size === maximum && !reached) {
        reached = true;
        // Do not await abort here: it waits for the loop emitting this event.
        void session.abort();
      }
    }
  });
  return { get calls() { return admitted.size; }, get reached() { return reached; } };
}
