export const SECTIONS_MAP = {
  cad_lab:              { display_name: 'CAD Lab',              max_capacity: 18 },
  focused_reading_area: { display_name: 'Focused Reading Area', max_capacity: 27 },
  g_hall_2:             { display_name: 'General Hall 2',       max_capacity: 92 },
  g_huss:               { display_name: 'G. Huss Reading Hall', max_capacity: 96 },
  hindi_section:        { display_name: 'Hindi Section',        max_capacity: 84 },
  ip_camera_19:         { display_name: 'Reading Lounge',       max_capacity: 11 },
  ipc:                  { display_name: 'IPC Computer Lab',     max_capacity: 16 },
  main_computer_room:   { display_name: 'Main Computer Room',   max_capacity: 15 },
  reference_2:          { display_name: 'Reference Section 2',  max_capacity: 24 },
  reference_area:       { display_name: 'Reference Area',       max_capacity: 13 },
  weeding_out_area:     { display_name: 'Weeding Out Area',     max_capacity: 12 },
};

export const TOTAL_LIBRARY_CAPACITY = 428;

export function getDisplayName(sectionId) {
  return SECTIONS_MAP[sectionId]?.display_name || sectionId;
}

export function getMaxCapacity(sectionId) {
  return SECTIONS_MAP[sectionId]?.max_capacity || 0;
}
