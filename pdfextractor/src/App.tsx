import { useState } from 'react'
import Layout from './components/Layout'
import EquipmentManagement from './components/EquipmentsManagement'

function App() {
    const [activeView, setActiveView] = useState('pdf-extractor')

    return (
        <Layout activeView={activeView} onViewChange={setActiveView}>
            <EquipmentManagement />
        </Layout>
    )
}

export default App
