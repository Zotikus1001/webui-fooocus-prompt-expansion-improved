# Fooocus Prompt Expansion Improved

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: CC-By NC 4.0](https://img.shields.io/badge/License-CC--By%20NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

An enhanced Stable Diffusion WebUI extension that automatically expands your prompts using Fooocus's AI-powered prompt expansion technology. Transform simple prompts into detailed, high-quality descriptions for better image generation results.

**Tested only on [SD WebUI Forge Classic](https://github.com/Haoming02/sd-webui-forge-classic)**

## ✨ Features

- **🤖 AI-Powered Expansion**: Automatically enhances prompts with relevant artistic and quality tags
- **⚡ Smart Caching**: LRU cache system prevents re-generation of identical prompts
- **🎯 Flexible Control**: Configurable expansion weights, tag limits, and seed control
- **🔧 Device Options**: CPU or CUDA acceleration support
- **🎲 Live Preview**: Test expansions before applying to your prompts
- **📊 Debug Mode**: Detailed console logging for troubleshooting

## 🚀 Installation

1. Open Stable Diffusion WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste this URL:
   ```
   https://github.com/Zotikus1001/z-webui-fooocus-prompt-expansion-improved.git
   ```
4. Click **Install**
5. Restart WebUI (model downloads automatically on first use)

## 📖 Usage

1. **Enable the extension** in the WebUI interface
2. **Enter your prompt** in the standard prompt box
3. **Generate images** - prompts are automatically expanded during generation
4. **Optional**: Use "Generate Preview" to test expansions before generating images

### Settings

- **Expansion Seed**: Control randomness (-1=random, 0=use generation seed, or fixed number)
- **Expansion Weight**: Apply weights to added tags (1.0=no weight, other values add emphasis)
- **Max Expansion Tags**: Limit number of added tags (0=unlimited)
- **Device Mode**: Choose CPU (stable) or CUDA (faster)
- **Debug Mode**: Enable detailed console logging

## 🔧 How It Works

The extension processes your prompts after any wildcards are resolved, using Fooocus's GPT-2 based model to intelligently add artistic style, quality, and technical tags that enhance image generation without changing your original intent.

**Example transformation:**
```
Input:  "a cat sitting"
Output: "a cat sitting, detailed fur, soft lighting, photorealistic, high quality, sharp focus, artistic composition"
```

## 📜 License

- **AGPL-3.0** for use within Fooocus ecosystem
- **CC-BY-NC-4.0** for non-commercial use outside Fooocus

## 🙏 Acknowledgements

- **lllyasviel** - Creator of the original Fooocus prompt expansion module
- **Stable Diffusion Community** - For continuous support and feedback
- **Contributors** - Everyone who helped improve this extension
