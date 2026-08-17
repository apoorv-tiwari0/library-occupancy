import React from 'react';
import { useLiveTime } from '../hooks/useLiveTime';
import { getStatus, getStatusColor, getStatusLabel } from '../utils/status';
import styles from './SectionCard.module.css';

export function SectionCard({ section }) {
  const {
    display_name,
    headcount = 0,
    max_capacity = 1,
    vacancy,
    occupancy_pct = 0,
    timestamp,
  } = section || {};

  const timeAgo = useLiveTime(timestamp);
  
  const freeSeats = typeof vacancy === 'number' ? vacancy : Math.max(0, max_capacity - headcount);
  const roundedPct = Math.round(occupancy_pct);
  const statusKey = getStatus(occupancy_pct);
  const statusLabel = getStatusLabel(occupancy_pct);
  const statusColor = getStatusColor(occupancy_pct);

  // Clamp percentage fill for gauge bar (0 to 100)
  const fillWidth = Math.max(0, Math.min(100, occupancy_pct));

  return (
    <article className={styles.card} aria-label={`${display_name}: ${roundedPct}% occupied`}>
      <div className={styles.topRow}>
        <div className={styles.statusBadge} style={{ color: statusColor }}>
          <span className={styles.statusDotSmall} style={{ backgroundColor: statusColor }} />
          <span>{statusLabel}</span>
        </div>
        <span className={styles.topRightDot} style={{ backgroundColor: statusColor }} />
      </div>

      <h2 className={styles.sectionTitle}>{display_name}</h2>

      <div className={styles.gaugeContainer}>
        <div className={styles.gaugeTrack}>
          <div
            className={styles.gaugeFill}
            style={{
              width: `${fillWidth}%`,
              backgroundColor: statusColor,
            }}
          />
        </div>
        <span className={styles.fraction}>
          {headcount}/{max_capacity}
        </span>
      </div>

      <div className={styles.secondaryInfo}>
        {roundedPct}% occupied &middot; {freeSeats} seat{freeSeats === 1 ? '' : 's'} free
      </div>

      <div className={styles.timestamp}>
        Last seen {timeAgo}
      </div>
    </article>
  );
}
