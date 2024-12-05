import axios from 'axios';
import { useState } from 'react';
import Button from './Button';

export default function FileUploader() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isData,setIsData] = useState(false);
  const [disorderType,setDisorderType] = useState('');

  function handleFileChange(e) {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  }

  async function handleFileUpload() {
    if (!file) return;

    setStatus('uploading');
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res= await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = progressEvent.total
            ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
            : 0;
          setUploadProgress(progress);
        },
      });

      setStatus('success');
      setUploadProgress(100);

      setIsData(true);
      setDisorderType(res.data.output)

      console.log("res",res)
    } catch {
      setStatus('error');
      setUploadProgress(0);
    }
  }

  return (
    <div className="space-y-2 flex flex-col">
        <div className='flex items-center justify-center mb-4'>
            <label
                htmlFor="file-upload"
                className="font-medium w-[100%] h-10 bg-black rounded-lg text-white text-center pt-2 cursor-pointer hover:bg-gray-600"
            >
                {file ? "Upload Another Patient's Details" :"Upload Patient's Details"}
            </label>
            <input
                type="file"
                id="file-upload"
                onChange={handleFileChange}
                className="hidden"
            />
        </div>


      {file && (
        <div className="mb-4 text-md text-white">
          <p>File name: {file.name}</p>
          <p>Size: {(file.size / 1024).toFixed(2)} KB</p>
          <p>Type: {file.type}</p>
        </div>
      )}


      {status === 'uploading' && (
        <div className="space-y-2">
          <div className="h-2.5 w-full rounded-full bg-gray-200">
            <div
              className="h-2.5 rounded-full bg-blue-600 transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <p className="text-md text-white">{uploadProgress}% uploaded</p>
        </div>
      )}

      {file && status !== 'uploading' && (
        <Button onClick={handleFileUpload} text="Analyse"/>
      )}

      <div className="text-white font-semibold text-lg pt-10">
        {isData?`Patient has ${disorderType}`:null}
      </div>
      {/* {status === 'success' && (
        <p className="text-sm text-green-600">File uploaded successfully!</p>
      )}

      {status === 'error' && (
        <p className="text-sm text-red-600">Upload failed. Please try again.</p>
      )} */}
    </div>
  );
}