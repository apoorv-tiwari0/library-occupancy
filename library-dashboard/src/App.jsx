import React from 'react';
import './styles/global.css';
import { useOccupancy } from './hooks/useOccupancy';
import { Header } from './components/Header';
import { SummaryBar } from './components/SummaryBar';
import { ConnectionBanner } from './components/ConnectionBanner';
import { SectionGrid } from './components/SectionGrid';

export default function App() {
  const {
    sections,
    summary,
    isLoading,
    isError,
    errorMessage,
    wsStatus,
    lastSystemUpdate,
    refetch,
  } = useOccupancy();

  return (
    <div className="appContainer">
      <Header wsStatus={wsStatus} lastSystemUpdate={lastSystemUpdate} />
      <SummaryBar summary={summary} />
      <ConnectionBanner wsStatus={wsStatus} />
      <SectionGrid
        sections={sections}
        isLoading={isLoading}
        isError={isError}
        errorMessage={errorMessage}
        onRetry={refetch}
      />
    </div>
  );
}
