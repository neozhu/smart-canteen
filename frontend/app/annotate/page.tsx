"use client";

import { useEffect, useState, useRef } from "react";
import useSWR from "swr";
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

interface ClassInfo {
  name: string;
  count: number;
  samples: string[];
}

interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: number[]; // [x1, y1, x2, y2]
  is_default?: boolean;
}

interface PreviewData {
  success: boolean;
  frame_width: number;
  frame_height: number;
  detections: Detection[];
}

const fetcher = (url: string) => axios.get(url).then((res) => res.data);

export default function AnnotationPage() {
  const [classes, setClasses] = useState<string[]>([]);
  const [selectedClass, setSelectedClass] = useState<string>("");
  const [annotationStats, setAnnotationStats] = useState<Record<string, ClassInfo>>({});
  const [isCapturing, setIsCapturing] = useState(false);
  const [message, setMessage] = useState("");
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [showDetectionBoxes, setShowDetectionBoxes] = useState(true);
  
  const videoRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch classes
  const { data: classesData } = useSWR(`${API_BASE}/api/config/classes`, fetcher);

  // Fetch annotation stats
  const { data: statsData, mutate: refreshStats } = useSWR(
    `${API_BASE}/api/annotation/stats`,
    fetcher,
    { refreshInterval: 2000 }
  );

  // Fetch detection preview for annotation assistance
  const { data: previewData } = useSWR<PreviewData>(
    showDetectionBoxes ? `${API_BASE}/api/annotation/preview` : null,
    fetcher,
    { refreshInterval: 300, revalidateOnFocus: false }
  );

  const buildCenterSquareFromPreview = () => {
    if (!previewData) return null;
    const side = Math.min(previewData.frame_width, previewData.frame_height) * 0.8;
    const x1 = (previewData.frame_width - side) / 2;
    const y1 = (previewData.frame_height - side) / 2;
    return [x1, y1, x1 + side, y1 + side];
  };

  useEffect(() => {
    if (classesData) {
      setClasses(classesData);
      if (!selectedClass && classesData.length > 0) {
        setSelectedClass(classesData[0]);
      }
    }
  }, [classesData]);

  useEffect(() => {
    if (statsData) {
      setAnnotationStats(statsData);
    }
  }, [statsData]);

  // Draw detection boxes on canvas
  useEffect(() => {
    if (!showDetectionBoxes || !previewData || !previewData.success) {
      // Clear canvas if boxes disabled or no data
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Get actual display dimensions
    const rect = video.getBoundingClientRect();
    const displayWidth = rect.width;
    const displayHeight = rect.height;
    
    // Set canvas resolution to match display size (for sharp rendering)
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (previewData.detections.length === 0) return;

    // Calculate the actual video content area within the container
    // Video uses object-contain, so we need to calculate letterbox/pillarbox offsets
    const frameAspect = previewData.frame_width / previewData.frame_height;
    const displayAspect = displayWidth / displayHeight;
    
    let videoWidth, videoHeight, offsetX, offsetY;
    
    if (displayAspect > frameAspect) {
      // Pillarbox (black bars on sides)
      videoHeight = displayHeight;
      videoWidth = displayHeight * frameAspect;
      offsetX = (displayWidth - videoWidth) / 2;
      offsetY = 0;
    } else {
      // Letterbox (black bars on top/bottom)
      videoWidth = displayWidth;
      videoHeight = displayWidth / frameAspect;
      offsetX = 0;
      offsetY = (displayHeight - videoHeight) / 2;
    }

    // Calculate scaling factors for the actual video content
    const scaleX = videoWidth / previewData.frame_width;
    const scaleY = videoHeight / previewData.frame_height;

    // Draw each detection box
    previewData.detections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const scaledX1 = x1 * scaleX + offsetX;
      const scaledY1 = y1 * scaleY + offsetY;
      const scaledX2 = x2 * scaleX + offsetX;
      const scaledY2 = y2 * scaleY + offsetY;
      const width = scaledX2 - scaledX1;
      const height = scaledY2 - scaledY1;

      // Choose color based on confidence
      const isDefault = det.is_default;
      let color = "rgba(0, 128, 255, 0.8)"; // Blue for default/fallback box
      if (!isDefault) {
        color = "rgba(255, 0, 0, 0.6)"; // Red for very low confidence
        if (det.confidence > 0.5) {
          color = "rgba(0, 255, 0, 0.8)"; // Green for high confidence
        } else if (det.confidence > 0.25) {
          color = "rgba(255, 255, 0, 0.7)"; // Yellow for medium confidence
        }
      }

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(scaledX1, scaledY1, width, height);

      // Draw label background
      const label = det.is_default
        ? "默认中心框"
        : `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.font = "14px Arial";
      const textMetrics = ctx.measureText(label);
      const textHeight = 20;
      
      ctx.fillStyle = color;
      ctx.fillRect(scaledX1, scaledY1 - textHeight, textMetrics.width + 8, textHeight);

      // Draw label text
      ctx.fillStyle = "white";
      ctx.fillText(label, scaledX1 + 4, scaledY1 - 5);
    });
  }, [previewData, showDetectionBoxes]);

  const handleCapture = async () => {
    if (!selectedClass) {
      setMessage("⚠️ 请先选择标签类型");
      return;
    }

    setIsCapturing(true);
    setMessage("📸 正在捕获图像...");

    try {
      // Get the best matching detection bbox if available
      let bbox = null;
      if (previewData && previewData.detections && previewData.detections.length > 0) {
        // Find detection matching selected class
        const classIndex = classes.indexOf(selectedClass);
        const matchingDet = previewData.detections.find(
          (det) => det.class_id === classIndex
        );
        
        if (matchingDet) {
          bbox = matchingDet.bbox;
          setMessage("📸 使用检测到的边框标注...");
        } else if (previewData.detections.length > 0) {
          // Use first detection if no class match
          bbox = previewData.detections[0].bbox;
          setMessage("📸 使用第一个检测框标注...");
        }
      }

      // If still no bbox, fallback to centered square to keep annotations consistent
      if (!bbox) {
        const fallbackBox = buildCenterSquareFromPreview();
        if (fallbackBox) {
          bbox = fallbackBox;
          setMessage("📸 未检测到物体，使用中心默认框标注...");
        }
      }

      if (!bbox) {
        setMessage("📸 未获取检测框，将使用后端默认中心框标注...");
      }

      const response = await axios.post(`${API_BASE}/api/annotation/capture`, {
        label: selectedClass,
        bbox: bbox,
      });

      if (bbox) {
        setMessage(`✅ 已保存 ${selectedClass} 的样本 (使用检测框，共 ${response.data.total_count} 张)`);
      } else {
        setMessage(`✅ 已保存 ${selectedClass} 的样本 (自动检测边框，共 ${response.data.total_count} 张)`);
      }
      refreshStats();
    } catch (error) {
      console.error("Capture failed:", error);
      setMessage("❌ 捕获失败，请重试");
    } finally {
      setIsCapturing(false);
    }
  };

  const handleDeleteSample = async (label: string, filename: string) => {
    try {
      await axios.delete(`${API_BASE}/api/annotation/sample`, {
        data: { label, filename },
      });
      setMessage(`🗑️ 已删除样本 ${filename}`);
      refreshStats();
    } catch (error) {
      setMessage("❌ 删除失败");
    }
  };

  const handleStartTraining = async () => {
    const totalSamples = Object.values(annotationStats).reduce(
      (sum, info) => sum + info.count,
      0
    );

    if (totalSamples < 20) {
      setMessage("⚠️ 样本数量不足，建议每个类别至少 10 张图片");
      return;
    }

    const confirmed = confirm(
      `确认开始训练？\n总样本数: ${totalSamples}\n预计耗时: 5-30 分钟`
    );

    if (!confirmed) return;

    setIsTraining(true);
    setTrainingProgress(0);
    setMessage("🚀 训练启动中...");

    try {
      // Start training
      const response = await axios.post(`${API_BASE}/api/training/start`);
      const taskId = response.data.task_id;

      // Poll training progress
      const interval = setInterval(async () => {
        try {
          const progressRes = await axios.get(
            `${API_BASE}/api/training/progress/${taskId}`
          );
          const { progress, status, message: msg } = progressRes.data;

          setTrainingProgress(progress);
          setMessage(`⏳ ${msg || "训练中..."} (${progress}%)`);

          if (status === "completed") {
            clearInterval(interval);
            setIsTraining(false);
            setMessage("🎉 模型训练完成！系统将自动重载模型");
            setTimeout(() => {
              window.location.href = "/";
            }, 3000);
          } else if (status === "failed") {
            clearInterval(interval);
            setIsTraining(false);
            setMessage("❌ 训练失败: " + msg);
          }
        } catch (error) {
          clearInterval(interval);
          setIsTraining(false);
          setMessage("❌ 训练过程出错");
        }
      }, 2000);
    } catch (error) {
      setIsTraining(false);
      setMessage("❌ 启动训练失败");
    }
  };

  const getTotalSamples = () => {
    return Object.values(annotationStats).reduce((sum, info) => sum + info.count, 0);
  };

  const getClassProgress = (className: string) => {
    const count = annotationStats[className]?.count || 0;
    const target = 20; // Recommended samples per class
    return Math.min(100, (count / target) * 100);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">数据标注 & 模型训练</h1>
          <a
            href="/"
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            返回主页
          </a>
        </div>

        {message && (
          <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-blue-800">{message}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Feed & Capture */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">实时预览</h2>
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={showDetectionBoxes}
                  onChange={(e) => setShowDetectionBoxes(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm">显示检测框</span>
              </label>
            </div>

            <div className="relative bg-black rounded overflow-hidden mb-4 mx-auto" style={{ aspectRatio: '4 / 3', maxWidth: '640px' }}>
              <img
                ref={videoRef}
                src={`${API_BASE}/video_feed`}
                alt="Camera Feed"
                className="w-full h-full object-contain"
                onLoad={() => {
                  // Trigger canvas redraw when image loads
                  if (canvasRef.current && videoRef.current) {
                    const rect = videoRef.current.getBoundingClientRect();
                    canvasRef.current.width = rect.width;
                    canvasRef.current.height = rect.height;
                  }
                }}
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 pointer-events-none"
                style={{ 
                  display: showDetectionBoxes ? "block" : "none",
                  width: '100%',
                  height: '100%'
                }}
              />
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  选择标签类型
                </label>
                <select
                  value={selectedClass}
                  onChange={(e) => setSelectedClass(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                  disabled={isTraining}
                >
                  {classes.map((cls) => (
                    <option key={cls} value={cls}>
                      {cls} ({annotationStats[cls]?.count || 0} 张)
                    </option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleCapture}
                disabled={isCapturing || isTraining || !selectedClass}
                className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-300 text-white font-bold py-3 px-4 rounded-lg transition"
              >
                {isCapturing ? "捕获中..." : "📸 拍照标注"}
              </button>

              <div className="p-4 bg-gray-50 rounded border">
                <p className="text-sm text-gray-600">
                  💡 <strong>使用说明：</strong>
                </p>
                <ol className="text-sm text-gray-600 mt-2 space-y-1 list-decimal list-inside">
                  <li>将物品放置在摄像头前</li>
                  <li>勾选"显示检测框"可查看当前模型识别的位置</li>
                  <li>选择对应的标签类型</li>
                  <li>点击"拍照标注"保存样本</li>
                  <li>建议每个类别采集 20+ 张不同角度的照片</li>
                </ol>
                {previewData && showDetectionBoxes && (
                  <div className="mt-2 p-2 bg-blue-50 rounded text-xs">
                    {previewData.detections.some((det) => det.is_default) ? (
                      <p className="font-semibold text-blue-800">🎯 未检测到物体，已显示中心默认框</p>
                    ) : (
                      <>
                        <p className="font-semibold text-blue-800">
                          🎯 当前检测到 {previewData.detections.length} 个物体
                        </p>
                        {previewData.detections.slice(0, 3).map((det, idx) => (
                          <p key={idx} className="text-blue-700">
                            • {det.class_name}: {(det.confidence * 100).toFixed(1)}%
                          </p>
                        ))}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Statistics & Training */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold mb-4">标注统计</h2>

            <div className="mb-6 p-4 bg-blue-50 rounded">
              <div className="text-2xl font-bold text-blue-600">
                {getTotalSamples()} 张
              </div>
              <div className="text-sm text-gray-600">总样本数</div>
            </div>

            <div className="space-y-3 mb-6">
              {classes.map((cls) => (
                <div key={cls} className="border rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">{cls}</span>
                    <span className="text-sm text-gray-600">
                      {annotationStats[cls]?.count || 0} 张
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all"
                      style={{ width: `${getClassProgress(cls)}%` }}
                    ></div>
                  </div>
                  {annotationStats[cls]?.samples && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {annotationStats[cls].samples.slice(0, 5).map((sample) => (
                        <button
                          key={sample}
                          onClick={() => handleDeleteSample(cls, sample)}
                          className="text-xs px-2 py-1 bg-red-100 text-red-600 rounded hover:bg-red-200"
                          disabled={isTraining}
                        >
                          {sample.slice(0, 8)}... ×
                        </button>
                      ))}
                      {annotationStats[cls].samples.length > 5 && (
                        <span className="text-xs text-gray-500 px-2 py-1">
                          +{annotationStats[cls].samples.length - 5} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {isTraining ? (
              <div className="space-y-3">
                <div className="w-full bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-blue-500 h-4 rounded-full transition-all text-xs text-white text-center leading-4"
                    style={{ width: `${trainingProgress}%` }}
                  >
                    {trainingProgress}%
                  </div>
                </div>
                <p className="text-center text-sm text-gray-600">
                  训练中，请勿关闭页面...
                </p>
              </div>
            ) : (
              <button
                onClick={handleStartTraining}
                disabled={getTotalSamples() < 10}
                className="w-full bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white font-bold py-3 px-4 rounded-lg transition"
              >
                🚀 开始训练模型
              </button>
            )}

            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm">
              <p className="font-semibold text-yellow-800 mb-1">⚠️ 训练建议</p>
              <ul className="text-yellow-700 space-y-1 list-disc list-inside">
                <li>每个类别至少 20 张样本</li>
                <li>多角度、不同光照条件拍摄</li>
                <li>训练时间约 5-30 分钟</li>
                <li>训练过程中保持网络连接</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
