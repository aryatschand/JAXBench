# JAXBench Setup Guide

This guide covers setting up:
1. **Modal** - Cloud TPU allocation
2. **AWS Bedrock** - Claude Opus 4.5 API access

---

## AWS Bedrock (Claude Opus 4.5)

### Credentials

```
AWS Access Key ID:     REDACTED_AWS_ACCESS_KEY
AWS Secret Access Key: REDACTED_AWS_SECRET_KEY
AWS Region:            us-east-2
```

### Available Models

| Model | Alias | Model ID |
|-------|-------|----------|
| Claude Opus 4.5 | `opus` | `us.anthropic.claude-opus-4-5-20251101-v1:0` |
| Claude Haiku 4.5 | `haiku` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Claude Sonnet 3.5 | `sonnet` | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Qwen3 Coder 480B | `qwen-480b` | `qwen.qwen3-coder-480b-a35b-v1:0` |
| Qwen3 Coder 30B | `qwen-30b` | `qwen.qwen3-coder-30b-a3b-v1:0` |

### Quick Start

```python
from bedrock_client import BedrockClient

# Create client (uses hardcoded credentials)
client = BedrockClient()

# Use Claude Opus 4.5
response = client.invoke("opus", "Explain JAX in one paragraph.")
print(response)

# Use with system prompt
response = client.invoke(
    "opus",
    "Write a JAX function to compute softmax.",
    system="You are an expert JAX developer.",
    temperature=0.7,
    max_tokens=1000,
)
```

### Environment Variables (Optional Override)

If you want to use different credentials:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-2"
```

### Test Connection

```bash
python bedrock_client.py
```

---

## Modal Account Setup

### 1. Create Modal Account

Go to [https://modal.com](https://modal.com) and sign up for an account.

### 2. Get API Credentials

1. Go to [https://modal.com/settings](https://modal.com/settings)
2. Click on "API Tokens" or "Tokens"
3. Create a new token
4. Copy both:
   - **Token ID** (starts with `ak-`)
   - **Token Secret** (long string)

### 3. Configure Modal CLI

**Option A: Interactive Authentication (Recommended)**

```bash
pip install modal
modal token new
```

This opens a browser for authentication. Follow the prompts.

**Option B: Environment Variables**

Set these in your shell:

```bash
export MODAL_TOKEN_ID="ak-your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"
```

Or add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# Modal credentials
export MODAL_TOKEN_ID="ak-your-token-id"
export MODAL_TOKEN_SECRET="your-token-secret"
```

**Option C: Create a `.env` file**

Create a file called `.env` in this directory:

```
MODAL_TOKEN_ID=ak-your-token-id
MODAL_TOKEN_SECRET=your-token-secret
```

Then load it:
```bash
source .env  # or use python-dotenv
```

### 4. Verify Setup

```bash
# Check authentication
modal token show

# Run test
modal run tpu_runner.py
```

---

## TPU Access on Modal

Modal provides access to Google Cloud TPUs. Available types:

| TPU | Accelerator String | Cores | Memory | Cost |
|-----|-------------------|-------|--------|------|
| v4-8 | `tpu-v4-8` | 8 | 32GB HBM | $ |
| v4-16 | `tpu-v4-16` | 16 | 64GB HBM | $$ |
| v4-32 | `tpu-v4-32` | 32 | 128GB HBM | $$$ |
| v5e-4 | `tpu-v5e-4` | 4 | 16GB HBM | $ |
| v5e-8 | `tpu-v5e-8` | 8 | 32GB HBM | $$ |

### Request TPU Access

If you don't have TPU quota:
1. Contact Modal support
2. Request TPU access for your workspace
3. Wait for approval (usually 1-2 business days)

---

## Quick Test

Once authenticated:

```bash
# Test TPU connection
modal run tpu_runner.py

# Run a benchmark
modal run tpu_runner.py --benchmark matmul --size 4096

# Run attention benchmark
modal run tpu_runner.py --benchmark attention --size 1024
```

---

## Troubleshooting

### "No TPU quota"
- Contact Modal support to request TPU access
- Try a different TPU type

### "Authentication failed"
```bash
modal token new  # Re-authenticate
```

### "JAX not finding TPU"
- Ensure you're using `jax[tpu]` in the Modal image
- Check the libtpu installation

### "Timeout errors"
- Increase timeout in function decorator
- Check Modal dashboard for logs

