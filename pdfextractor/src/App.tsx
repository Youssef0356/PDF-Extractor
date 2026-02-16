import { useState } from 'react'
import Layout from './components/Layout'
import PDFDropZone from './components/PDFDropZone'

function App() {
    const [activeView, setActiveView] = useState('upload')

    const handleFileSelect = (file: File) => {
        console.log('File selected:', file)
    }

    return (
        <Layout activeView={activeView} onViewChange={setActiveView}>
            <div className="flex flex-col items-center justify-center h-full">
                <h1 className="text-2xl font-bold mb-8 text-gray-800">Upload PDF Document</h1>
                <PDFDropZone onFileSelect={handleFileSelect} />
            </div>
        </Layout>
    )
}

export default App
