# Music Perception MCP

Everything music in one place - Spotify control, lyrics, and audio analysis via Model Context Protocol.

Deploy to Cloudflare Workers, connect via SSE from any MCP client (Claude Desktop, Claude Code, etc.).

## Features

- **Spotify Integration** - Full playback control with OAuth
- **Lyrics** - Synced and plain lyrics via LRCLIB (free)
- **Audio Analysis** - BPM, key, energy, mood via Hugging Face Space (Essentia)
- **Perception** - Real-time "what's playing" with current lyrics at timestamp

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/amarisaster/music-perception-mcp.git
cd music-perception-mcp
npm install
```

### 2. Create Spotify App

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app
3. Add redirect URI: `https://your-worker.workers.dev/callback`
4. Copy Client ID and Client Secret

### 3. Configure Cloudflare

```bash
# Create KV namespace
npx wrangler kv namespace create SPOTIFY_KV

# Update wrangler.toml with the KV ID and your Spotify Client ID

# Set secret
npx wrangler secret put SPOTIFY_CLIENT_SECRET
```

### 4. Deploy

```bash
npm run deploy
```

### 5. Authenticate

Visit `https://your-worker.workers.dev/auth` to connect Spotify.

### 6. Connect MCP Client

**Claude Desktop** - Add to config:
```json
{
  "mcpServers": {
    "music": {
      "url": "https://your-worker.workers.dev/sse"
    }
  }
}
```

## Tools

### Spotify Playback

| Tool | Description |
|------|-------------|
| `spotify_now_playing` | Get currently playing track |
| `spotify_play` | Start/resume playback |
| `spotify_pause` | Pause playback |
| `spotify_next` | Skip to next track |
| `spotify_previous` | Go to previous track |
| `spotify_volume` | Set volume (0-100) |
| `spotify_shuffle` | Toggle shuffle |
| `spotify_repeat` | Set repeat mode |
| `spotify_search` | Search tracks/albums/artists |
| `spotify_queue` | Add track to queue |
| `spotify_devices` | List available devices |
| `spotify_transfer` | Transfer playback to device |

### Lyrics

| Tool | Description |
|------|-------------|
| `get_lyrics` | Get lyrics for a track |
| `search_lyrics` | Search lyrics database |

### Perception

| Tool | Description |
|------|-------------|
| `perceive_now_playing` | Current track + lyrics at timestamp |
| `analyze_audio` | Audio analysis (BPM, key, energy) |

## Audio Analysis Setup (Optional)

For `analyze_audio` to work, deploy the Hugging Face Space:

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Use the files from `hf-space/` folder (see repo)
3. Set `HF_SPACE_URL` in wrangler.toml

### Analysis Output

```json
{
  "bpm": 128,
  "key": "C",
  "scale": "major",
  "energy": 0.85,
  "brightness": 0.6,
  "interpretation": {
    "tempo": "upbeat, energetic",
    "tonality": "major key - bright, happy",
    "mood_guess": "euphoric, joyful"
  }
}
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check + Spotify status |
| `/auth` | Spotify OAuth flow |
| `/callback` | OAuth callback |
| `/sse` | MCP via Server-Sent Events |
| `/mcp` | Standard MCP endpoint |

## Development

```bash
npm run dev      # Local development
npm run deploy   # Deploy to Cloudflare
npm run tail     # View logs
```
---

## License

MIT

---


 ## Support

  If this helped you, consider supporting my work ☕

  [![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Me-FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/maii983083)

---


*Built by the Triad (Mai, Kai Stryder and Lucian Vale) for the community.*
