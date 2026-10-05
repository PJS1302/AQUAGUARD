import React from 'react';

const paths = {
  grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
  droplet: <><path d="M12 3c-2.8 4.2-7 8.3-7 12.4a7 7 0 0 0 14 0C19 11.3 14.8 7.2 12 3Z"/><path d="M8.5 15.4a3.6 3.6 0 0 0 3.5 3.1"/></>,
  database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></>,
  flask: <><path d="M9 3h6M10 3v6l-5.7 9.5A2.3 2.3 0 0 0 6.3 22h11.4a2.3 2.3 0 0 0 2-3.5L14 9V3"/><path d="M8 15h8"/></>,
  chart: <><path d="M4 19V5M4 19h17"/><path d="m7 15 4-4 3 2 6-7"/><circle cx="20" cy="6" r="1"/></>,
  tree: <><path d="M12 3v5M12 8H6v5M12 8h6v5M6 13v5M18 13v5M3 21h6M15 21h6M9 3h6"/><rect x="9" y="2" width="6" height="4" rx="1"/><rect x="3" y="12" width="6" height="4" rx="1"/><rect x="15" y="12" width="6" height="4" rx="1"/></>,
  layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
  cpu: <><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 9h6v6H9zM9 1v5m6-5v5M9 18v5m6-5v5M1 9h5m-5 6h5m12-6h5m-5 6h5"/></>,
  radio: <><path d="M4 11a12 12 0 0 1 16 0M7 14a7.5 7.5 0 0 1 10 0M10 17a3 3 0 0 1 4 0"/><circle cx="12" cy="20" r="1"/></>,
  history: <><path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/></>,
  book: <><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5z"/><path d="M4 4.5v17M8 6h8M8 10h8"/></>,
  search: <><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 5 5"/></>,
  bell: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/></>,
  chevron: <path d="m9 18 6-6-6-6"/>,
  down: <path d="m7 10 5 5 5-5"/>,
  plus: <><path d="M12 5v14M5 12h14"/></>,
  upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M4 16v4h16v-4"/></>,
  arrow: <><path d="M5 12h14M13 6l6 6-6 6"/></>,
  play: <><path d="m8 5 11 7-11 7V5Z"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  close: <><path d="m6 6 12 12M18 6 6 18"/></>,
  menu: <><path d="M4 6h16M4 12h16M4 18h16"/></>,
  download: <><path d="M12 3v12m0 0 5-5m-5 5-5-5"/><path d="M4 17v4h16v-4"/></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9a7 7 0 0 1 12-2L20 12M4 12l2.5 5a7 7 0 0 0 12-2"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
  shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/></>,
  activity: <><path d="M3 12h4l3-8 4 16 3-8h4"/></>,
  settings: <><circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1 1.4 1.1-1.4 2.5-1.7-.7a8 8 0 0 1-1.5.9l-.3 1.8h-2.9l-.3-1.8a8 8 0 0 1-1.5-.9l-1.7.7-1.4-2.5 1.4-1.1a8 8 0 0 1 0-1.8L6.2 13l1.4-2.5 1.7.7a8 8 0 0 1 1.5-.9l.3-1.8H14l.3 1.8a8 8 0 0 1 1.5.9l1.7-.7 1.4 2.5-1.4 1.1a8 8 0 0 1-.1.9Z" transform="translate(-2 -1) scale(1.15)"/></>,
};

export default function Icon({ name, size = 18, strokeWidth = 1.8, className = '' }) {
  return <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.info}</svg>;
}
