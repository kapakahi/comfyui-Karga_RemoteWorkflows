# Karga Remote Workflow

A standalone ComfyUI custom node that sends workflows to a remote ComfyUI instance and returns the output image — all in a single node. Part of the Karga ecosystem, maintained as its own separate repository.

## Features

- **Workflow picker** — drop any `workflow_api.json` into the `workflows/` folder and select it from a dropdown
- **Remote execution** — queues the job on a remote ComfyUI instance over HTTP and polls until complete
- **Prompt & seed** — built-in prompt box and seed control (with ComfyUI's native randomize/fixed/increment)
- **Optional image input** — connect an image for img2img workflows, or leave it disconnected for text-to-image
- **Optional mask input** — connect a mask for inpainting workflows
- **Dynamic fields** — any additional `[ui]`-tagged nodes in your workflow automatically appear as inputs on the node

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/KargaRemoteWorkflow
```

Restart ComfyUI. The **Karga Remote Workflow** node will appear under the `Karga` category.

## Setup

1. Export your remote workflow using **Save (API Format)** in ComfyUI
2. Place the exported JSON in `custom_nodes/KargaRemoteWorkflow/workflows/`
3. Make sure your remote ComfyUI is running with `--listen`

## Usage

Add the **Karga Remote Workflow** node to your graph and configure:

| Input | Description |
|---|---|
| `workflow` | Select a workflow JSON from the `workflows/` folder |
| `remote_address` | Address of the remote ComfyUI instance, e.g. `192.168.1.50:8188` |
| `prompt` | Text prompt — injected into the workflow's `[ui]` prompt node |
| `noise_seed` | Seed — injected into the workflow's `[ui]` seed node |
| `image` *(optional)* | Input image for img2img workflows |
| `mask` *(optional)* | Mask for inpainting workflows |
| `poll_interval` | How often to check for completion (seconds) |
| `timeout` | Max wait time before raising an error (seconds) |
| `image_index` | Which output image to return if the workflow produces multiple |

## Tagging your remote workflow

Tag nodes in your remote workflow so this node knows where to inject values. Rename a node's title using either format:

**Suffix style** (recommended for standard nodes):
```
Load Image [ui]
CLIP Text Encode (Positive Prompt) [ui]
RandomNoise [ui]
```

**Prefix style** (for custom labels or non-standard nodes):
```
[ui] prompt:text
[ui] steps:steps
[ui] cfg:cfg
```

### Supported suffix-style class types

| Class type | Injected value | Input key |
|---|---|---|
| `LoadImage` | Uploaded input image filename | `image` |
| `LoadImageMask` | Uploaded mask filename | `image` |
| `CLIPTextEncode` | Prompt text | `text` |
| `RandomNoise` | Seed value | `noise_seed` |

Any other node type tagged with `[ui]` will fall back to using its first scalar input.

### Prefix-style format

```
[ui] label_name:input_field
```

Examples:
```
[ui] steps:steps
[ui] cfg:cfg
[ui] input_image:image
[ui] input_mask:image
```

Any prefix-style `[ui]` fields that aren't `prompt`, `noise_seed`, `input_image`, or `input_mask` will appear as additional input widgets on the node automatically.

## License

MIT
