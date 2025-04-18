import React, { useState, useCallback } from "react";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Predictor = () => {
  const [formData, setFormData] = useState({
    GRE: "",
    TOEFL: "",
    "University Rating": "",
    SOP: "",
    LOR: "",
    CGPA: "",
    Research: "",
  });
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState("");

  const API_URL = import.meta.env.VITE_EXP_API_URL ;

  const handleChange = (e) => {
    setFormData((prevFormData) => ({ ...prevFormData, [e.target.name]: e.target.value }));
  };

  const validateInput = () => {
    for (let key in formData) {
      if (formData[key] === "") {
        toast.error(`Please enter ${key}`);
        return false;
      }
    }
    return true;
  };

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!validateInput()) return;
    setLoading(true);
    try {
      const formattedData = Object.fromEntries(
        Object.entries(formData).map(([key, value]) => [key, parseFloat(value)])
      );
      const response = await axios.post(`${API_URL}/predict`, formattedData,{timeout:100000});
      setPrediction(response.data.admission_chance);
      toast.success("Prediction received!");
    } catch (error) {
      toast.error("Error fetching prediction");
    }
    setLoading(false);
  }, [formData]);

  const handleReset = () => {
    setFormData({
      GRE: "",
      TOEFL: "",
      "University Rating": "",
      SOP: "",
      LOR: "",
      CGPA: "",
      Research: "",
    });
    setPrediction("");
  };

  const fieldRanges = {
    GRE: { min: 260, max: 340 },
    TOEFL: { min: 0, max: 120 },
    "University Rating": { min: 1, max: 5 },
    SOP: { min: 1, max: 5 },
    LOR: { min: 1, max: 5 },
    CGPA: { min: 0, max: 10 },
    Research: { min: 0, max: 1 },
  };

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md p-6 shadow-lg bg-white rounded-2xl">
        <h2 className="text-2xl font-bold text-center text-orange-600 mb-4">
          UCLA Admission Predictor
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          {Object.keys(formData).map((field) => (
            <div key={field}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {field} (Range: {fieldRanges[field]?.min} - {fieldRanges[field]?.max})
              </label>
              <input
                type="number"
                step="any"
                name={field}
                value={formData[field]}
                onChange={handleChange}
                placeholder={`Enter ${field}`}
                className="p-2 border rounded-md w-full"
                min={fieldRanges[field]?.min}
                max={fieldRanges[field]?.max}
              />
            </div>
          ))}
          <div className="flex justify-between space-x-2">
            <button
              type="submit"
              className={`w-1/2 bg-orange-600 hover:bg-orange-700 text-white p-2 rounded-md 
              ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
              disabled={loading}
            >
              {loading ? "Predicting..." : "Get Prediction"}
            </button>
            <button
              type="button"
              className="w-1/2 bg-gray-400 hover:bg-gray-500 text-white p-2 rounded-md"
              onClick={handleReset}
            >
              Reset
            </button>
          </div>
        </form>
        {prediction !== "" && (
          <div className="text-center mt-4">
            <p className="text-lg font-semibold text-black">Chance of Admission:</p>
            <p className="text-xl font-bold text-orange-600">{prediction}%</p>
          </div>
        )}
      </div>
      <ToastContainer />
    </div>
  );
};

export default Predictor;
