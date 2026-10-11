# RadioScribe

Underground mine radio traffic monitor with offline speech-to-text and keyword alerting.

**© Shawn Baird. All rights reserved. Non-commercial use only.**

---

## Overview

RadioScribe monitors radio traffic through your phone's microphone, transcribes speech offline, and alerts you when critical keywords are detected. Built for underground mining environments.

## Alert Tiers

| Level | Trigger | Color | Behavior |
|-------|---------|-------|----------|
| Normal | No match | White | Scrolls off naturally |
| Warning | Keyword match | industrial Yellow | Pinned, swipe to dismiss |
| Safety | Safety keyword | tool handle Red | Pinned top, alarm after 20s |

## Architecture

```
lib/
  main.dart               # App entry point
  theme/
    app_theme.dart        # Colors (industrial yellow, tool red, dark bg)
  models/
    segment.dart          # TranscriptSegment data model
    keyword_config.dart   # Keyword lists + settings model
  services/
    audio_service.dart    # Mic input + Vosk STT (Vosk hooks here)
    keyword_service.dart  # Keyword/phrase matching engine
    settings_service.dart # Persistent settings via SharedPreferences
    transcript_log_service.dart  # Optional text log per session
  screens/
    splash_screen.dart    # Disclaimer + license
    main_menu_screen.dart # Main menu
    listen_screen.dart    # Live monitoring view
    settings_screen.dart  # Keyword editor + preferences
  widgets/
    segment_card.dart     # Swipeable segment display card
```

## License

Non-commercial use only. All rights reserved. © Shawn Baird.
