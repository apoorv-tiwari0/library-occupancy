import { useState, useEffect } from 'react';
import { formatTimeAgo } from '../utils/status';

export function useLiveTime(timestamp) {
  const [timeAgo, setTimeAgo] = useState(() => formatTimeAgo(timestamp));

  useEffect(() => {
    setTimeAgo(formatTimeAgo(timestamp));

    const interval = setInterval(() => {
      setTimeAgo(formatTimeAgo(timestamp));
    }, 1000);

    return () => clearInterval(interval);
  }, [timestamp]);

  return timeAgo;
}
