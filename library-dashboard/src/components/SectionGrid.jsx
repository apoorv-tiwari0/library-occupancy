import React from 'react';
import { SectionCard } from './SectionCard';
import { SkeletonCard } from './SkeletonCard';
import styles from './SectionGrid.module.css';

export function SectionGrid({ sections, isLoading, isError, errorMessage, onRetry }) {
  // Render 11 skeletons during initial load
  if (isLoading) {
    return (
      <main className={styles.gridContainer}>
        <div className={styles.grid}>
          {Array.from({ length: 11 }).map((_, index) => (
            <SkeletonCard key={index} />
          ))}
        </div>
      </main>
    );
  }

  return (
    <main className={styles.gridContainer}>
      <div className={styles.grid}>
        {isError && (
          <div className={styles.errorState} role="alert">
            <span className={styles.errorText}>
              {errorMessage || 'Unable to load occupancy data. Retrying...'}
            </span>
            {onRetry && (
              <button type="button" className={styles.retryButton} onClick={onRetry}>
                Retry Fetch
              </button>
            )}
          </div>
        )}

        {sections.map((section) => (
          <SectionCard key={section.section_id} section={section} />
        ))}
      </div>
    </main>
  );
}
