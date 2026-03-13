import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// V2 Data Interfaces
// ---------------------------------------------------------------------------

export interface InstrumentData {
  category?: string;
  typeMesure?: string;
  typeActionneur?: string;
  code?: string;
  technologie?: string;
  codeTechnologie?: string;
  plageMesureMin?: number;
  plageMesureMax?: number;
  plageMesureUnite?: string;
  signalSortie?: string;
  hart?: boolean;
  nombreFils?: number;
  alimentation?: string;
  communication?: string;
  sortieTOR?: boolean;
  seuil?: number;
  seuilUnite?: string;
  repérageArmoire?: string;
  precision?: string;
  marque?: string;
  référence?: string;
  certificats?: string[];
  indiceIP?: string;
  températureProcess?: string;
  matériauMembrane?: string;
  datasheetUrl?: string;
  typeVérin?: string;
  typeActionneurSpécial?: string;
  positionSécurité?: string;
  courseMM?: number;
  forceN?: number;
  pressionAlimentationBar?: number;
}

export interface ExtractionResponse {
  success: boolean;
  data?: InstrumentData;
  confidence?: Record<string, number>;
  message?: string;
  processing_time_seconds?: number;
  meta?: {
    pdf_id: string;
  };
  doc_context?: {
    category: string;
    code?: string;
    typeMesure?: string;
    source: string;
  };
}

export interface CorrectionRecord {
  field_name: string;
  ai_extracted_value: any;
  user_corrected_value: any;
  rule?: string;
  accepted: boolean;
}

export interface CorrectionBatch {
  pdf_id: string;
  category?: string;
  typeMesure?: string;
  corrections: CorrectionRecord[];
}

export interface SchemaOptions {
  categories: string[];
  typesMesure: string[];
  typesActionneur: string[];
  codes: Record<string, string[]>;
  technologies: Record<string, string[]>;
  signals: string[];
  powers: string[];
  communications: string[];
  brands: string[];
  indicesIP: string[];
  materials: string[];
  actuatorTypes: string[];
  safetyPositions: string[];
  specialActuatorTypes: string[];
}

// ---------------------------------------------------------------------------
// API Methods
// ---------------------------------------------------------------------------

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const extractFromPDF = async (file: File): Promise<ExtractionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<ExtractionResponse>('/extract', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const submitCorrections = async (batch: CorrectionBatch): Promise<any> => {
  const response = await api.post('/corrections', batch);
  return response.data;
};

export const fetchSchemaOptions = async (): Promise<SchemaOptions> => {
  const response = await api.get<SchemaOptions>('/schema/options');
  return response.data;
};

export const checkApiHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default {
  extractFromPDF,
  submitCorrections,
  fetchSchemaOptions,
  checkApiHealth,
};
