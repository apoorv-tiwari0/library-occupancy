import React from 'react';
import styles from './ConnectionBanner.module.css';

export function ConnectionBanner({ wsStatus }) {
  if (wsStatus === 'connected') {
    return null;
  }

  return (
    <div className={styles.banner} role="status" aria-live="polite">
      <span className={styles.iconDot} />
      <span>Live updates paused — reconnecting...</span>
    </div>
  );
}
