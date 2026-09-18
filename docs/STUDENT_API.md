# Student gateway route

The optional `gpt_student` runtime in `configs/formal_training.toml` uses
`https://flowsteer.org:2087/v1`, model `lab-gpt-5.5-2`, Responses API,
direct networking, no streaming, concurrency 10, and a 600-second timeout.
Existing GPT routes and the default Worker route pool are unchanged. Select
`gpt_student` explicitly when running an experiment with this gateway.

Credentials are read from `~/.config/student-api/flowsteer.key`; the parent
directory must be private and the key file should have mode 600. Never commit
the key or include it in commands, logs, or experiment artifacts.

`responses_text` encodes local action schemas as conversation text and tool
observations as user messages. It does not send native `tools` or function-call
items. Worker runtime parses returned action JSON using its existing parser.
This differs from native tool calling and needs task-level evaluation before
comparisons with other routes. Automatic HTTP inference retries are disabled
for this profile; explicit later agent calls are separate requests.

Initial verification: authenticated models listing contained the selected
model. One project-backend inference returned a parseable echo action, using
129 input tokens and 20 output tokens. No benchmark was run.
