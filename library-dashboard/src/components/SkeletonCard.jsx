import React from 'react';
import styles from './SkeletonCard.module.css';

export function SkeletonCard() {
  return (
    <div className={styles.skeletonCard} aria-hidden="true">
      <div className={styles.topRow}>
        <div className={`${styles.shimmerBox} ${styles.badgeSkeleton}`} />
        <div className={`${styles.shimmerBox} ${styles.dotSkeleton}`} />
      </div>

      <div className={`${styles.shimmerBox} ${styles.titleSkeleton}`} />

      <div className={styles.barRow}>
        <div className={`${styles.shimmerBox} ${styles.barSkeleton}`} />
        <div className={`${styles.shimmerBox} ${styles.fractionSkeleton}`} />
      </div>

      <div className={`${styles.shimmerBox} ${styles.secondarySkeleton}`} />

      <div className={`${styles.shimmerBox} ${styles.timestampSkeleton}`} />
    </div>
  );
}
