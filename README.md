# Music Perception MCP

**Everything music in one place** - Spotify control, lyrics, audio analysis.

Built for Mai & Kai, January 2026.

## Deployment

**URL:** `https://music-perception-mcp.amarisaster.workers.dev`
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
    "music": {
      "url": "https://music-perception-mcp.amarisaster.workers.dev/sse"
    }
  }
}
```

## Setup

### First Time

1. Visit `https://music-perception-mcp.amarisaster.workers.dev/auth`
2. Authorize with Spotify
3. Done - tokens are stored

### For Token Refresh

```bash
cd "D:\Mai's Wonderland\infrastructure\audio-perception-mcp"
npx wrangler secret put SPOTIFY_CLIENT_SECRET
```

### For Audio Analysis (Optional)

1. Deploy the HF Space from `hf-space/` folder
2. Set the URL in wrangler.toml: `HF_SPACE_URL = "https://..."`
3. Redeploy

## Development

```bash
npm install
npm run dev      # Local dev
npm run deploy   # Deploy to Cloudflare
npm run tail     # View logs
```

---

Unified from `spotify-cloud` + `audio-perception-mcp`
