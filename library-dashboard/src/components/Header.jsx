import React from 'react';
import { useLiveTime } from '../hooks/useLiveTime';
import styles from './Header.module.css';

export function Header({ wsStatus, lastSystemUpdate }) {
  const timeAgo = useLiveTime(lastSystemUpdate);

  const getStatusText = () => {
    if (wsStatus === 'connected') return 'Live';
    if (wsStatus === 'reconnecting') return 'Reconnecting...';
    return 'Offline';
  };

  const getDotClass = () => {
    if (wsStatus === 'connected') return `${styles.pulseDot} ${styles.pulseDotConnected}`;
    if (wsStatus === 'reconnecting') return `${styles.pulseDot} ${styles.pulseDotReconnecting}`;
    return styles.pulseDot;
  };

  return (
    <header className={styles.header}>
      <div className={styles.leftGroup}>
        <span className={styles.title}>IIT Delhi Library</span>
        <span className={styles.subtitle}>Occupancy Monitor</span>
      </div>

      <div className={styles.rightGroup}>
        <div className={styles.liveIndicator}>
          <span className={getDotClass()} />
          <span>{getStatusText()}</span>
        </div>
        <span className={styles.lastUpdated}>
          Updated {timeAgo}
        </span>
      </div>
    </header>
  );
}
