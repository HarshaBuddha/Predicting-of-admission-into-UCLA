import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Predictor from './components/predictorform'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
     <Predictor/>
    </>
  )
}

export default App
