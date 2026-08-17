import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const httpClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

export const fetchSections = async () => {
  const response = await httpClient.get('/sections');
  return response.data;
};

export const fetchSectionById = async (sectionId) => {
  const response = await httpClient.get(`/sections/${sectionId}`);
  return response.data;
};

export const fetchSectionHistory = async (sectionId, hours = 24) => {
  const response = await httpClient.get(`/sections/${sectionId}/history`, {
    params: { hours },
  });
  return response.data;
};
