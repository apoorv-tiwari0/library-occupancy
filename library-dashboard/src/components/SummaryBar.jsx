import React, { useEffect, useState } from 'react';
import styles from './SummaryBar.module.css';

function AnimatedNumber({ value }) {
  const [displayValue, setDisplayValue] = useState(value);

  useEffect(() => {
    let startTimestamp = null;
    const startValue = displayValue;
    const targetValue = value;
    const duration = 400; // 0.4s ease

    if (startValue === targetValue) return;

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      // Ease out quad formula
      const easedProgress = progress * (2 - progress);
      const current = Math.round(startValue + (targetValue - startValue) * easedProgress);
      setDisplayValue(current);

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    };

    const animFrame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animFrame);
  }, [value]);

  return <span>{displayValue}</span>;
}

export function SummaryBar({ summary }) {
  const {
    totalCapacity = 241,
    totalOccupied = 0,
    totalAvailable = 241,
    openSectionsCount = 0,
    totalSections = 11,
  } = summary || {};

  return (
    <section className={styles.summaryContainer} aria-label="Library occupancy summary">
      <div className={styles.summaryContent}>
        <div className={styles.statItem}>
          <span className={styles.label}>Total Seats</span>
          <span className={styles.value}>
            <AnimatedNumber value={totalCapacity} />
          </span>
        </div>

        <div className={styles.divider} />

        <div className={styles.statItem}>
          <span className={styles.label}>Occupied</span>
          <span className={styles.value}>
            <AnimatedNumber value={totalOccupied} />
          </span>
        </div>

        <div className={styles.divider} />

        <div className={styles.statItem}>
          <span className={styles.label}>Available</span>
          <span className={styles.value}>
            <AnimatedNumber value={totalAvailable} />
          </span>
        </div>

        <div className={styles.divider} />

        <div className={styles.statItem}>
          <span className={styles.label}>Sections Open</span>
          <span className={styles.value}>
            <AnimatedNumber value={openSectionsCount} />
            <span style={{ fontSize: '18px', fontWeight: '400', color: 'var(--color-text-secondary)' }}>
              /{totalSections}
            </span>
          </span>
        </div>
      </div>
    </section>
  );
}
