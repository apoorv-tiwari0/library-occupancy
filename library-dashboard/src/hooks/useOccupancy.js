import { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchSections } from '../api/http';
import { wsManager } from '../api/websocket';
import { SECTIONS_MAP, TOTAL_LIBRARY_CAPACITY } from '../utils/sections';

// Helper to normalize backend object or fallback default
function normalizeSection(rawSection) {
  const sectionId = rawSection.section_id;
  const config = SECTIONS_MAP[sectionId] || {
    display_name: sectionId || 'Unknown Section',
    max_capacity: rawSection.max_capacity || 20,
  };

  const headcount = typeof rawSection.headcount === 'number' ? rawSection.headcount : 0;
  const maxCapacity = rawSection.max_capacity || config.max_capacity;
  const vacancy = typeof rawSection.vacancy === 'number' ? rawSection.vacancy : Math.max(0, maxCapacity - headcount);
  const occupancyPct = typeof rawSection.occupancy_pct === 'number'
    ? rawSection.occupancy_pct
    : Number(((headcount / (maxCapacity || 1)) * 100).toFixed(1));

  return {
    section_id: sectionId,
    display_name: rawSection.display_name || config.display_name,
    headcount,
    max_capacity: maxCapacity,
    vacancy,
    occupancy_pct: occupancyPct,
    is_available: typeof rawSection.is_available === 'boolean' ? rawSection.is_available : headcount < maxCapacity,
    timestamp: rawSection.timestamp || new Date().toISOString(),
    inference_ms: rawSection.inference_ms || 0,
    pipeline_ms: rawSection.pipeline_ms || 0,
  };
}

// Build initial section collection for all 11 defined sections
function createDefaultSectionsMap() {
  const initialMap = {};
  Object.entries(SECTIONS_MAP).forEach(([id, meta]) => {
    initialMap[id] = {
      section_id: id,
      display_name: meta.display_name,
      headcount: 0,
      max_capacity: meta.max_capacity,
      vacancy: meta.max_capacity,
      occupancy_pct: 0,
      is_available: true,
      timestamp: new Date().toISOString(),
      inference_ms: 0,
      pipeline_ms: 0,
    };
  });
  return initialMap;
}

export function useOccupancy() {
  const [sectionsMap, setSectionsMap] = useState(createDefaultSectionsMap);
  const [isLoading, setIsLoading] = useState(true);
  const [isError, setIsError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [lastSystemUpdate, setLastSystemUpdate] = useState(new Date().toISOString());

  const loadInitialData = useCallback(async () => {
    setIsLoading(true);
    setIsError(false);
    setErrorMessage('');
    try {
      const data = await fetchSections();
      if (Array.isArray(data)) {
        setSectionsMap((prev) => {
          const next = { ...prev };
          let newestTime = null;
          data.forEach((item) => {
            if (item.section_id) {
              const normalized = normalizeSection(item);
              next[item.section_id] = normalized;
              if (!newestTime || new Date(normalized.timestamp) > new Date(newestTime)) {
                newestTime = normalized.timestamp;
              }
            }
          });
          if (newestTime) setLastSystemUpdate(newestTime);
          return next;
        });
      }
    } catch (err) {
      console.error('[useOccupancy] Error fetching sections:', err);
      setIsError(true);
      setErrorMessage(err.response?.data?.detail || err.message || 'Unable to load occupancy data.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch initial REST data & connect WS
  useEffect(() => {
    loadInitialData();

    // Connect WebSocket
    wsManager.connect();

    // Subscribe to WS connection status
    const unsubscribeStatus = wsManager.subscribeStatus((state) => {
      setWsStatus(state);
    });

    // Subscribe to WS data updates
    const unsubscribeData = wsManager.subscribeData((msg) => {
      if (msg && (msg.type === 'occupancy_update' || msg.section_id) && (msg.data || msg.section_id)) {
        const item = msg.data || msg;
        if (item.section_id) {
          const normalized = normalizeSection(item);
          setSectionsMap((prev) => ({
            ...prev,
            [normalized.section_id]: normalized,
          }));
          setLastSystemUpdate(normalized.timestamp || new Date().toISOString());
        }
      }
    });

    return () => {
      unsubscribeStatus();
      unsubscribeData();
      wsManager.disconnect();
    };
  }, [loadInitialData]);

  // Derived sorted sections list: Occupancy % descending (busiest first)
  const sortedSections = useMemo(() => {
    return Object.values(sectionsMap).sort((a, b) => b.occupancy_pct - a.occupancy_pct);
  }, [sectionsMap]);

  // Derived summary numbers
  const summary = useMemo(() => {
    const sectionList = Object.values(sectionsMap);
    const totalOccupied = sectionList.reduce((sum, sec) => sum + (sec.headcount || 0), 0);
    const totalAvailable = Math.max(0, TOTAL_LIBRARY_CAPACITY - totalOccupied);
    const openSectionsCount = sectionList.filter((sec) => sec.is_available !== false).length;
    const totalSections = sectionList.length;

    return {
      totalCapacity: TOTAL_LIBRARY_CAPACITY,
      totalOccupied,
      totalAvailable,
      openSectionsCount,
      totalSections,
    };
  }, [sectionsMap]);

  return {
    sections: sortedSections,
    summary,
    isLoading,
    isError,
    errorMessage,
    wsStatus,
    lastSystemUpdate,
    refetch: loadInitialData,
  };
}
