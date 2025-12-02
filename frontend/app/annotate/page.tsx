"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
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
  }, [classesData, selectedClass]);

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
      setMessage("请先选择标签类型");
      return;
    }

    setIsCapturing(true);
    setMessage("正在捕获图像...");

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
          setMessage("使用检测到的边框标注...");
        } else if (previewData.detections.length > 0) {
          // Use first detection if no class match
          bbox = previewData.detections[0].bbox;
          setMessage("使用第一个检测框标注...");
        }
      }

      // If still no bbox, fallback to centered square to keep annotations consistent
      if (!bbox) {
        const fallbackBox = buildCenterSquareFromPreview();
        if (fallbackBox) {
          bbox = fallbackBox;
          setMessage("未检测到物体，使用中心默认框标注...");
        }
      }

      if (!bbox) {
        setMessage("未获取检测框，将使用后端默认中心框标注...");
      }

      const response = await axios.post(`${API_BASE}/api/annotation/capture`, {
        label: selectedClass,
        bbox: bbox,
      });

      if (bbox) {
        setMessage(`已保存 ${selectedClass} 的样本 (使用检测框，共 ${response.data.total_count} 张)`);
      } else {
        setMessage(`已保存 ${selectedClass} 的样本 (自动检测边框，共 ${response.data.total_count} 张)`);
      }
      refreshStats();
    } catch (error) {
      console.error("Capture failed:", error);
      setMessage("捕获失败，请重试");
    } finally {
      setIsCapturing(false);
    }
  };

  const handleDeleteSample = async (label: string, filename: string) => {
    try {
      await axios.delete(`${API_BASE}/api/annotation/sample`, {
        data: { label, filename },
      });
      setMessage(`已删除样本 ${filename}`);
      refreshStats();
    } catch (error) {
      console.error("Delete failed:", error);
      setMessage("删除失败");
    }
  };

  const handleStartTraining = async () => {
    const totalSamples = Object.values(annotationStats).reduce(
      (sum, info) => sum + info.count,
      0
    );

    if (totalSamples < 20) {
      setMessage("样本数量不足，建议每个类别至少 10 张图片");
      return;
    }

    const confirmed = confirm(
      `确认开始训练？\n总样本数: ${totalSamples}\n预计耗时: 5-30 分钟`
    );

    if (!confirmed) return;

    setIsTraining(true);
    setTrainingProgress(0);
    setMessage("训练启动中...");

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
          setMessage(`${msg || "训练中..."} (${progress}%)`);

          if (status === "completed") {
            clearInterval(interval);
            setIsTraining(false);
            setMessage("模型训练完成，系统将自动重载模型");
            setTimeout(() => {
              window.location.href = "/";
            }, 3000);
          } else if (status === "failed") {
            clearInterval(interval);
            setIsTraining(false);
            setMessage("训练失败: " + msg);
          }
        } catch (error) {
          console.error("Training poll failed:", error);
          clearInterval(interval);
          setIsTraining(false);
          setMessage("训练过程中出错");
        }
      }, 2000);
    } catch (error) {
      console.error("Start training failed:", error);
      setIsTraining(false);
      setMessage("启动训练失败");
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
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-gray-100 text-slate-900">
      <div className="max-w-7xl mx-auto px-6 py-10 space-y-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 text-sm font-semibold text-orange-700 shadow-sm ring-1 ring-orange-100">
              <span className="h-2.5 w-2.5 rounded-full bg-orange-500 animate-pulse" />
              现代清新·标注工作台
            </div>
            <div className="space-y-1">
              <h1 className="text-3xl font-bold tracking-tight">数据标注 & 模型训练</h1>
              <p className="text-gray-600 max-w-2xl">
                使用暖橙主色与中性灰背景，保持高对比度的触屏体验，让标注、校准与训练都更加沉浸易用。
              </p>
            </div>
          </div>
          <Link
            href="/"
            className="inline-flex items-center gap-2 rounded-xl bg-white/80 px-4 py-2 text-sm font-semibold text-slate-700 shadow-md shadow-orange-100/70 ring-1 ring-slate-200 transition hover:-translate-y-0.5 hover:shadow-lg"
          >
            返回主页
          </Link>
        </div>

        {message && (
          <div className="rounded-xl border border-orange-200 bg-orange-50/70 px-4 py-3 text-sm font-medium text-orange-800 shadow-sm">
            {message}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Video Feed & Capture */}
          <div className="space-y-5 rounded-2xl bg-white/95 p-6 shadow-[0_28px_80px_-40px_rgba(17,24,39,0.4)] ring-1 ring-slate-100">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-orange-700">实时预览</p>
                <p className="text-lg font-bold text-slate-900">辅助标注</p>
                <p className="text-sm text-gray-500">对齐检测框，捕获高质量样本</p>
              </div>
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-full bg-orange-50 px-3 py-2 text-xs font-medium text-orange-700 ring-1 ring-orange-100">
                <input
                  type="checkbox"
                  checked={showDetectionBoxes}
                  onChange={(e) => setShowDetectionBoxes(e.target.checked)}
                  className="h-4 w-4 rounded border-orange-300 text-orange-600 focus:ring-orange-500"
                />
                显示检测框
              </label>
            </div>

            <div
              className="relative mx-auto overflow-hidden rounded-2xl bg-slate-900/80 ring-1 ring-slate-200/70"
              style={{ aspectRatio: "4 / 3", maxWidth: "640px" }}
            >
              <img
                ref={videoRef}
                src={`${API_BASE}/video_feed`}
                alt="Camera Feed"
                className="h-full w-full object-contain"
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
                className="pointer-events-none absolute inset-0"
                style={{
                  display: showDetectionBoxes ? "block" : "none",
                  width: "100%",
                  height: "100%",
                }}
              />
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <label className="block text-sm font-semibold text-slate-800">选择标签类型</label>
                <select
                  value={selectedClass}
                  onChange={(e) => setSelectedClass(e.target.value)}
                  className="w-full rounded-xl border border-orange-200 bg-orange-50/60 px-3 py-2 text-slate-800 shadow-sm transition focus:border-orange-400 focus:outline-none focus:ring-2 focus:ring-orange-200 disabled:opacity-60"
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
                className="w-full rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 py-3 text-base font-bold text-white shadow-lg shadow-orange-200/80 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:from-slate-200 disabled:to-slate-200 disabled:text-slate-400"
              >
                {isCapturing ? "捕获中..." : "拍照标注"}
              </button>

              <div className="space-y-3 rounded-xl bg-slate-50 px-4 py-4 ring-1 ring-slate-100">
                <p className="text-sm font-semibold text-slate-800">使用说明</p>
                <ol className="list-decimal list-inside space-y-1 text-sm text-slate-600">
                  <li>将物品放置在摄像头前，保持光线柔和</li>
                  <li>开启“显示检测框”查看当前识别位置</li>
                  <li>选择对应的标签后点击“拍照标注”</li>
                  <li>建议每个类别采集 20+ 张不同角度的照片</li>
                </ol>
                {previewData && showDetectionBoxes && (
                  <div className="mt-2 space-y-1 rounded-lg bg-orange-50 px-3 py-2 text-xs text-orange-800 ring-1 ring-orange-100">
                    {previewData.detections.some((det) => det.is_default) ? (
                      <p className="font-semibold">未检测到物体，已显示中心默认框</p>
                    ) : (
                      <>
                        <p className="font-semibold">当前检测到 {previewData.detections.length} 个物体</p>
                        {previewData.detections.slice(0, 3).map((det, idx) => (
                          <p key={idx} className="text-orange-700">
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
          <div className="space-y-5 rounded-2xl bg-white/95 p-6 shadow-[0_28px_80px_-40px_rgba(17,24,39,0.4)] ring-1 ring-slate-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-orange-700">标注统计</p>
                <h2 className="text-xl font-bold text-slate-900">进度与样本</h2>
              </div>
              <div className="rounded-full bg-orange-50 px-3 py-1 text-xs font-semibold text-orange-700 ring-1 ring-orange-100">
                合计 {getTotalSamples()} 张
              </div>
            </div>

            <div className="rounded-xl bg-gradient-to-r from-orange-50 to-amber-50 px-4 py-4 shadow-inner ring-1 ring-orange-100">
              <p className="text-sm text-orange-700">高质量样本能显著提升模型准确率</p>
              <p className="text-2xl font-bold text-orange-600">{getTotalSamples()} 张</p>
              <p className="text-xs text-orange-700/80">实时统计 · 每 2s 自动刷新</p>
            </div>

            <div className="space-y-3">
              {classes.map((cls) => (
                <div key={cls} className="space-y-2 rounded-xl border border-slate-100 bg-white px-3 py-3 shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-800">{cls}</span>
                    <span className="text-sm text-slate-500">{annotationStats[cls]?.count || 0} 张</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-slate-100">
                    <div
                      className="h-2 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all"
                      style={{ width: `${getClassProgress(cls)}%` }}
                    ></div>
                  </div>
                  {annotationStats[cls]?.samples && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {annotationStats[cls].samples.slice(0, 5).map((sample) => (
                        <button
                          key={sample}
                          onClick={() => handleDeleteSample(cls, sample)}
                          className="rounded-full bg-orange-50 px-3 py-1 text-[11px] font-medium text-orange-700 ring-1 ring-orange-100 transition hover:bg-orange-100"
                          disabled={isTraining}
                        >
                          {sample.slice(0, 8)}... ×
                        </button>
                      ))}
                      {annotationStats[cls].samples.length > 5 && (
                        <span className="rounded-full bg-slate-100 px-3 py-1 text-[11px] text-slate-500">
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
                <div className="h-4 w-full rounded-full bg-slate-100">
                  <div
                    className="h-4 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 text-center text-xs font-semibold text-white leading-4 shadow-sm"
                    style={{ width: `${trainingProgress}%` }}
                  >
                    {trainingProgress}%
                  </div>
                </div>
                <p className="text-center text-sm text-slate-500">训练中，请勿关闭页面...</p>
              </div>
            ) : (
              <button
                onClick={handleStartTraining}
                disabled={getTotalSamples() < 10}
                className="w-full rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 py-3 text-base font-bold text-white shadow-lg shadow-orange-200/80 transition hover:scale-[1.01] disabled:cursor-not-allowed disabled:from-slate-200 disabled:to-slate-200"
              >
                开始训练模型
              </button>
            )}

            <div className="rounded-xl bg-yellow-50 px-4 py-3 text-sm text-yellow-800 ring-1 ring-yellow-200">
              <p className="font-semibold mb-1">训练建议</p>
              <ul className="list-disc list-inside space-y-1 text-yellow-700">
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
