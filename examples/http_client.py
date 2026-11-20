"""Call the running proxy over HTTP with the OpenAI SDK.

1. start it:  agent-proxy serve --offline        (port 8710, bundled demo policy)
2. run this:  python examples/http_client.py
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8710/v1",
    api_key="billing-demo-key",  # public demo key from the bundled policy
    default_headers={"X-Agent-Id": "billing-agent", "X-Session-Id": "case-12"},
)
reply = client.chat.completions.create(
    model="mock",
    messages=[{"role": "user", "content": "Customer giulia@example.com asks for invoice INV-2291. "
               'lookup_invoice(invoice_id="INV-2291")'}],
    tools=[{"type": "function", "function": {"name": "lookup_invoice", "parameters": {"type": "object"}}}],
)
print(reply.choices[0].message)
print(reply.model_extra.get("agent_proxy") if reply.model_extra else None)
