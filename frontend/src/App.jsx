import { useState } from 'react'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(e.target.files[0])
      setAnalysisResult(null)
      setError("")
    }
  }

  const handleAnalyze = async () => {
    if (!selectedImage) return;
    setLoading(true);
    setError("");
    
    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const res = await fetch("http://localhost:8080/api/v1/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Failed to analyze image");
      }

      const data = await res.json();
      setAnalysisResult(data);
    } catch (err) {
      console.error(err);
      setError("An error occurred during analysis or the backend is offline. (Demo mode fallback)");
      // Provide a mock demo result if backend is offline to satisfy the 'demo mode' requirement
      setAnalysisResult({
        crop: "Rice (Demo)",
        disease: "Brown Spot (Demo)",
        disease_confidence: 0.89,
        severity: "Medium",
        affected_area_percentage: 15.4,
        pests: [{class_name: "rice_bug", confidence: 0.92, bbox: [10, 10, 50, 50]}],
        recommendation: "Demo Recommendation: Apply appropriate fungicide.",
        processing_time_ms: 120,
        explainability_image: null
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans">
      <header className="bg-agri-dark text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            🌱 AgriGuard
          </h1>
          <nav className="flex gap-4">
            <button className="hover:text-agri-light font-medium transition-colors">Dashboard</button>
            <button className="hover:text-agri-light font-medium transition-colors">Models</button>
            <button className="hover:text-agri-light font-medium transition-colors">About</button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 mt-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* Upload Section */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col items-center">
            <h2 className="text-xl font-semibold mb-4 w-full text-left text-gray-700">Image Analysis</h2>
            
            <div className="w-full border-2 border-dashed border-gray-300 rounded-lg p-8 flex flex-col items-center justify-center bg-gray-50 hover:bg-gray-100 transition-colors cursor-pointer relative">
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleImageChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              {!selectedImage ? (
                <div className="text-center text-gray-500">
                  <svg className="w-12 h-12 mx-auto mb-2 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                  </svg>
                  <p>Click or drag image to upload</p>
                </div>
              ) : (
                <img 
                  src={URL.createObjectURL(selectedImage)} 
                  alt="Selected" 
                  className="max-h-64 object-contain rounded"
                />
              )}
            </div>

            <button 
              onClick={handleAnalyze}
              disabled={!selectedImage || loading}
              className={`mt-6 w-full py-3 rounded-lg font-bold text-white transition-all
                ${!selectedImage ? 'bg-gray-300 cursor-not-allowed' : 'bg-agri-green hover:bg-agri-dark shadow-md'}`}
            >
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </button>
            {error && <p className="mt-4 text-red-500 text-sm w-full text-center">{error}</p>}
          </div>

          {/* Results Section */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-full">
            <h2 className="text-xl font-semibold mb-4 text-gray-700">Analysis Results</h2>
            
            {!analysisResult ? (
              <div className="h-64 flex items-center justify-center text-gray-400">
                Upload and analyze an image to see results here.
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Crop Detected</p>
                    <p className="text-lg font-semibold text-agri-dark">{analysisResult.crop}</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Disease Identified</p>
                    <p className="text-lg font-semibold text-red-600">{analysisResult.disease}</p>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-500">Confidence Score</p>
                    <p className="text-lg font-semibold">{(analysisResult.disease_confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">Processing Time</p>
                    <p className="text-lg font-semibold">{analysisResult.processing_time_ms} ms</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Severity</p>
                    <p className={`text-lg font-semibold 
                      ${analysisResult.severity === 'Low' ? 'text-green-600' : 
                        analysisResult.severity === 'Medium' ? 'text-orange-500' : 'text-red-600'}`}>
                      {analysisResult.severity}
                    </p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500">Affected Area</p>
                    <p className="text-lg font-semibold">{analysisResult.affected_area_percentage}%</p>
                  </div>
                </div>
                
                {analysisResult.pests && analysisResult.pests.length > 0 && (
                  <div className="bg-red-50 border border-red-100 p-4 rounded-lg">
                    <p className="text-sm text-red-800 font-bold mb-1">Pests Detected!</p>
                    <ul className="list-disc pl-5 text-sm text-red-700">
                      {analysisResult.pests.map((p, i) => (
                        <li key={i}>{p.class_name} ({(p.confidence * 100).toFixed(1)}%)</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="bg-green-50 border border-green-100 p-4 rounded-lg mt-4">
                  <p className="text-sm font-bold text-green-800 mb-1">Actionable Recommendation</p>
                  <p className="text-sm text-green-700">{analysisResult.recommendation}</p>
                </div>

              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
