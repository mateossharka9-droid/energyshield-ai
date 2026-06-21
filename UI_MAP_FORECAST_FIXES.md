# EnergyShield NTL - UI, Map and Forecast Fixes

This version keeps the data/model pipeline structure but improves the professional demo experience.

## UI
- Rebuilt the Streamlit shell with a dark, harmonious operations sidebar inspired by the provided template.
- Removed the visible Streamlit collapsed-control text issue that showed `keyboard_double_arrow`.
- Added compact navigation icons and stronger active-state styling.
- Reduced horizontal overflow and table/card layout issues.

## GIS / Map
- Generated Albania coordinates now use inland-safe city centers and smaller jitter.
- Coastal cities are shifted slightly inland for prototype GIS so customer points do not appear in the sea.
- Map rendering and GeoJSON export now validate coordinates using an Albania-focused display envelope.

## Forecasting
- Replaced the previous simple linear forecast with a robust weekly-seasonal + recent-trend planning forecast.
- Loss forecast now has realistic variation, event-pressure logic, and a lower/upper planning range.
- Forecast chart separates suspicious loss from event counts using a secondary axis.

## Note
Forecast values remain operational planning signals, not legal fraud proof. In production, OSHEE would connect confirmed inspection outcomes, real GIS meter registry data, and live weather feeds.
