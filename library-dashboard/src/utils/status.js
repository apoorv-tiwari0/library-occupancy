export function getStatus(occupancyPct) {
  const pct = Math.max(0, Math.min(100, Number(occupancyPct) || 0));
  if (pct <= 40) return 'AVAILABLE';
  if (pct <= 70) return 'MODERATE';
  if (pct <= 90) return 'BUSY';
  return 'FULL';
}

export function getStatusLabel(occupancyPct) {
  const status = getStatus(occupancyPct);
  switch (status) {
    case 'AVAILABLE': return 'Available';
    case 'MODERATE': return 'Moderate';
    case 'BUSY': return 'Busy';
    case 'FULL': return 'Full';
    default: return 'Available';
  }
}

export function getStatusColor(occupancyPct) {
  const status = getStatus(occupancyPct);
  switch (status) {
    case 'AVAILABLE': return 'var(--status-available)';
    case 'MODERATE': return 'var(--status-moderate)';
    case 'BUSY': return 'var(--status-busy)';
    case 'FULL': return 'var(--status-full)';
    default: return 'var(--status-available)';
  }
}

export function getStatusHexColor(occupancyPct) {
  const status = getStatus(occupancyPct);
  switch (status) {
    case 'AVAILABLE': return '#2D6A4F';
    case 'MODERATE': return '#B5850A';
    case 'BUSY': return '#C0392B';
    case 'FULL': return '#7B2D2D';
    default: return '#2D6A4F';
  }
}

export function formatTimeAgo(timestamp) {
  if (!timestamp) return 'Just now';
  
  const date = new Date(timestamp);
  if (isNaN(date.getTime())) return 'Just now';

  const diffMs = Math.max(0, Date.now() - date.getTime());
  const diffSeconds = Math.floor(diffMs / 1000);

  if (diffSeconds < 2) return 'Just now';
  if (diffSeconds < 60) return `${diffSeconds}s ago`;
  
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  return `${Math.floor(diffHours / 24)}d ago`;
}
