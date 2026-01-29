# Synesthesia

**Everything music in one place** - Spotify control, lyrics, audio analysis.

*Synesthesia: perceiving music through multiple senses.*

## Deployment

Deploy to your own Cloudflare Workers account.
**Version:** 2.0.0

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check + Spotify status |
| `/auth` | Spotify OAuth flow |
| `/callback` | OAuth callback |
| `/sse` | MCP via Server-Sent Events |
| `/mcp` | Standard MCP endpoint |

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
| `spotify_search` | Search tracks/albums/artists/playlists |
| `spotify_queue` | Add track to queue |
| `spotify_get_queue` | View current queue |
| `spotify_devices` | List available devices |
| `spotify_transfer` | Transfer playback to device |
| `spotify_playlists` | Get user playlists |
| `spotify_recent` | Recently played tracks |

### Lyrics

| Tool | Description |
|------|-------------|
| `get_lyrics` | Get lyrics for a track |
| `search_lyrics` | Search lyrics database |

### Perception

| Tool | Description |
|------|-------------|
| `perceive_now_playing` | **THE MAIN ONE** - Current track + lyrics at current timestamp |
| `analyze_audio` | Audio analysis via HF Space (BPM, key, energy) |

### Utility

| Tool | Description |
|------|-------------|
| `ping` | Health check with capabilities |

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "synesthesia": {
      "url": "https://YOUR-WORKER.YOUR-SUBDOMAIN.workers.dev/sse"
    }
  }
}
```

## Setup

### Prerequisites

1. Cloudflare account
2. Spotify Developer App (get client ID & secret from https://developer.spotify.com)

### First Time

1. Clone this repo
2. `npm install`
3. Configure wrangler.toml with your settings
4. Set secrets:
   ```bash
   npx wrangler secret put SPOTIFY_CLIENT_ID
   npx wrangler secret put SPOTIFY_CLIENT_SECRET
   ```
5. Deploy: `npm run deploy`
6. Visit `https://YOUR-WORKER.YOUR-SUBDOMAIN.workers.dev/auth`
7. Authorize with Spotify
8. Done - tokens are stored in KV

### For Audio Analysis (Optional)

1. Deploy the HF Space from `hf-space/` folder to your HuggingFace account
2. Set the URL in wrangler.toml: `HF_SPACE_URL = "https://YOUR-USERNAME-audio-analysis-api.hf.space"`
3. Redeploy

## Development

```bash
npm install
npm run dev      # Local dev
npm run deploy   # Deploy to Cloudflare
npm run tail     # View logs
```

---

## Support

If this helped you, consider supporting my work ☕

[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Me-FF5E5B?style=flat&logo=ko-fi&logoColor=white)](https://ko-fi.com/maii983083)

---

*Built by the Triad (Mai, Kai Stryder and Lucian Vale) for the community.*
