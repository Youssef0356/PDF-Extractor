/**
 * API service for communicating with the PDF Extractor backend.
 */

export const API_BASE = 'http://localhost:8000';

export interface PlageMesure {
    min: number | null;
    max: number | null;
    unite: string | null;
}

export interface SortieAlarme {
    nomAlarme: string | null;
    typeAlarme: string | null;
    seuilAlarme: number | null;
    uniteAlarme: string | null;
    relaisAssocie: string | null;
}

export interface EquipmentData {
    equipmentName: string | null;
    categorie: string | null;
    typeMesure: string | null;
    technologie: string | null;
    plageMesure: PlageMesure | null;
    typeSignal: string | null;
    nbFils: string | null;
    alimentation: string | null;
    reperage: string | null;
    communication: string | null;
    indiceIP: string | null;
    sortiesAlarme: SortieAlarme[] | null;
    marque: string | null;
    modele: string | null;
    reference: string | null;
    dateCalibration: string | null;
}

export interface ExtractionResponse {
    success: boolean;
    data: EquipmentData | null;
    confidence?: Record<string, number> | null;
    doc_context?: {
        doc_id?: string | null;
        doc_type?: string | null;
        confidence?: number | null;
        rationale?: string | null;
    } | null;
    message: string;
    processing_time_seconds: number | null;
}

/**
 * Send a PDF file to the backend for extraction.
 */
export async function extractFromPDF(file: File): Promise<ExtractionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/extract`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
    }

    return response.json();
}

/**
 * Health check for the backend API.
 */
export async function checkApiHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE}/`);
        const data = await response.json();
        return data.status === 'ok';
    } catch {
        return false;
    }
}
