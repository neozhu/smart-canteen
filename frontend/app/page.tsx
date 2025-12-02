"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

interface Detection {
  label: string;
  confidence: number;
  bbox: number[];
}

interface CartItem {
  id: string;
  label: string;
  name: string;
  price: number;
  quantity: number;
}

interface PriceMapItem {
  name: string;
  price: number;
  category?: string;
  enabled?: boolean;
}

const fetcher = (url: string) => axios.get(url).then((res) => res.data);

export default function OperatorScreen() {
  const [cart, setCart] = useState<CartItem[]>([]);
  const [priceMap, setPriceMap] = useState<Record<string, PriceMapItem>>({});
  const [isDetecting, setIsDetecting] = useState(false);
  const [lastDetectionTime, setLastDetectionTime] = useState(0);

  // Don't use SWR for detection, use manual polling instead

  // Fetch price map
  const { data: priceMapData } = useSWR(
    `${API_BASE}/api/config/price_map`,
    fetcher,
    {
      revalidateOnFocus: false,
    }
  );

  useEffect(() => {
    if (priceMapData) {
      setPriceMap(priceMapData);
    }
  }, [priceMapData]);

  // Auto-detect every 1 second
  useEffect(() => {
    const detectAndUpdateCart = async () => {
      const now = Date.now();
      if (isDetecting || now - lastDetectionTime < 1000) return;

      setIsDetecting(true);
      setLastDetectionTime(now);

      try {
        const response = await axios.get(`${API_BASE}/api/detect_once`);
        const detections: Detection[] = response.data.detections || [];

        // Replace cart with current detections (not accumulate)
        if (detections.length === 0) {
          // No detection, clear cart
          setCart([]);
        } else {
          // Build new cart from detections
          const newCart: CartItem[] = [];
          const labelCounts = new Map<string, number>();

          // Count occurrences of each label
          detections.forEach((det) => {
            const count = labelCounts.get(det.label) || 0;
            labelCounts.set(det.label, count + 1);
          });

          // Create cart items
          labelCounts.forEach((quantity, label) => {
            const itemInfo = priceMap[label];
            if (itemInfo && itemInfo.enabled !== false) {
              newCart.push({
                id: `${label}_${now}`,
                label,
                name: itemInfo.name,
                price: itemInfo.price,
                quantity,
              });
            }
          });

          setCart(newCart);
        }
      } catch (error) {
        console.error("Detection failed:", error);
      } finally {
        setIsDetecting(false);
      }
    };

    // Run detection every 100ms, but internal logic ensures 1s interval
    const interval = setInterval(detectAndUpdateCart, 100);
    return () => clearInterval(interval);
  }, [isDetecting, lastDetectionTime, priceMap]);

  // Cart is now replaced entirely by detection results, no manual add

  const removeFromCart = (id: string) => {
    setCart((prev) => prev.filter((item) => item.id !== id));
  };

  const updateQuantity = (id: string, delta: number) => {
    setCart((prev) =>
      prev
        .map((item) =>
          item.id === id
            ? { ...item, quantity: Math.max(0, item.quantity + delta) }
            : item
        )
        .filter((item) => item.quantity > 0)
    );
  };

  const calculateTotal = (items: CartItem[] = cart) => {
    return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
  };

  const handleCheckout = async () => {
    const order = {
      items: cart,
      total: calculateTotal(),
      timestamp: new Date().toISOString(),
    };

    try {
      await axios.post(`${API_BASE}/api/order`, order);
      alert("结账成功!");
      setCart([]);
    } catch (error) {
      console.error("Checkout failed:", error);
      alert("结账失败,请重试");
    }
  };

  const broadcastCart = useCallback(
    (nextCart: CartItem[]) => {
      // Use BroadcastChannel to sync with customer screen
      if (typeof BroadcastChannel !== "undefined") {
        const channel = new BroadcastChannel("cart_sync");
        const total = nextCart.reduce(
          (sum, item) => sum + item.price * item.quantity,
          0
        );
        channel.postMessage({ cart: nextCart, total });
        channel.close();
      }
    },
    []
  );

  useEffect(() => {
    broadcastCart(cart);
  }, [broadcastCart, cart]);

  const totalItems = cart.reduce((sum, item) => sum + item.quantity, 0);

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-gray-100 text-slate-900">
      <div className="max-w-7xl mx-auto px-6 py-10 space-y-8">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 shadow-sm ring-1 ring-orange-100">
              <span className="h-2.5 w-2.5 rounded-full bg-orange-500 animate-pulse" />
              <span className="text-sm font-semibold text-orange-700">实时视觉结算</span>
            </div>
            <div className="space-y-2">
              <h1 className="text-4xl font-bold tracking-tight">Smart Canteen 操作员工作台</h1>
              <p className="text-gray-600 max-w-2xl">
                以暖橙为主色的现代清新界面，实时查看识别状态、调优标注，并为顾客提供顺畅的触屏体验。
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/customer"
              className="inline-flex items-center gap-2 rounded-xl bg-white/80 px-4 py-2 text-sm font-medium text-slate-700 shadow-md shadow-orange-100/60 ring-1 ring-gray-200 transition hover:-translate-y-0.5 hover:shadow-lg"
            >
              顾客显示屏
            </Link>
            <Link
              href="/annotate"
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-orange-200/80 transition hover:scale-[1.01]"
            >
              数据标注
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-xl bg-white/90 p-4 shadow-xl shadow-orange-100/50 ring-1 ring-orange-100">
            <p className="text-sm text-gray-500">识别状态</p>
            <div className="mt-2 flex items-center gap-2 text-lg font-semibold text-slate-900">
              <span
                className={`inline-flex h-2.5 w-2.5 rounded-full ${
                  isDetecting ? "bg-emerald-500 animate-pulse" : "bg-gray-400"
                }`}
              />
              {isDetecting ? "识别中" : "就绪"}
            </div>
            <p className="mt-1 text-xs text-gray-500">自动识别间隔：1 秒</p>
          </div>
          <div className="rounded-xl bg-white/90 p-4 shadow-xl shadow-orange-100/50 ring-1 ring-orange-100">
            <p className="text-sm text-gray-500">已识别菜品</p>
            <p className="mt-2 text-3xl font-bold text-slate-900">{totalItems}</p>
            <p className="mt-1 text-xs text-gray-500">购物车中商品数量</p>
          </div>
          <div className="rounded-xl bg-white/90 p-4 shadow-xl shadow-orange-100/50 ring-1 ring-orange-100">
            <p className="text-sm text-gray-500">金额合计</p>
            <p className="mt-2 text-3xl font-bold text-orange-600">¥{calculateTotal().toFixed(2)}</p>
            <p className="mt-1 text-xs text-gray-500">实时同步顾客屏</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Video Feed */}
          <div className="rounded-2xl bg-white/95 p-6 shadow-[0_28px_80px_-40px_rgba(17,24,39,0.4)] ring-1 ring-slate-100">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-orange-700">实时视频流</p>
                <p className="text-lg font-semibold text-slate-900">AI 识别预览</p>
                <p className="text-sm text-gray-500">保持菜品居中，提升识别稳定性</p>
              </div>
              <div className="rounded-full bg-orange-50 px-3 py-1 text-xs font-medium text-orange-700 ring-1 ring-orange-100">
                {isDetecting ? "运行中" : "等待识别"}
              </div>
            </div>

            <div
              className="relative mt-4 overflow-hidden rounded-2xl bg-slate-900/80 ring-1 ring-slate-200/70"
              style={{ aspectRatio: "4 / 3", maxWidth: "680px" }}
            >
              <img
                src={`${API_BASE}/video_feed`}
                alt="Camera Feed"
                className="h-full w-full object-contain"
              />
              <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-slate-900/70 to-transparent px-4 py-3 text-xs text-white">
                <span className="flex items-center gap-1">自动检测 · 1s/次</span>
                <span className="rounded-full bg-white/15 px-3 py-1 font-medium">{cart.length} 个检测结果</span>
              </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="flex items-center gap-3 rounded-xl bg-orange-50/60 px-4 py-3 text-sm text-orange-800 ring-1 ring-orange-100">
                <span className="text-lg" aria-hidden>提示</span>
                <div>
                  <p className="font-semibold">建议：保持镜头清洁</p>
                  <p className="text-xs text-orange-700/80">避免眩光，确保菜品居中摆放</p>
                </div>
              </div>
              <div className="flex items-center gap-3 rounded-xl bg-slate-50 px-4 py-3 text-sm text-slate-700 ring-1 ring-slate-100">
                <span className="text-lg" aria-hidden>同步</span>
                <div>
                  <p className="font-semibold">同步状态</p>
                  <p className="text-xs text-slate-500">变更将即时推送到顾客屏幕</p>
                </div>
              </div>
            </div>
          </div>

          {/* Cart */}
          <div className="rounded-2xl bg-white/95 p-6 shadow-[0_28px_80px_-40px_rgba(17,24,39,0.4)] ring-1 ring-slate-100">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-orange-700">购物车</p>
                <h2 className="text-xl font-bold text-slate-900">识别到的菜品</h2>
              </div>
              <span className="rounded-full bg-orange-50 px-3 py-1 text-xs font-semibold text-orange-700 ring-1 ring-orange-100">
                {cart.length} 类目
              </span>
            </div>

            {cart.length === 0 ? (
              <div className="mt-8 flex flex-col items-center rounded-xl bg-slate-50 px-4 py-10 text-center text-slate-500 ring-1 ring-slate-100">
                <span className="mb-3 text-3xl" aria-hidden>提示</span>
                <p className="font-semibold text-slate-700">等待识别结果</p>
                <p className="text-sm text-slate-500">菜品放入摄像范围后会自动填充购物车</p>
              </div>
            ) : (
              <div className="mt-4 space-y-3">
                {cart.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between rounded-xl bg-gradient-to-r from-white to-orange-50/60 p-4 shadow-sm ring-1 ring-orange-100"
                  >
                    <div className="flex-1">
                      <p className="text-base font-semibold text-slate-900">{item.name}</p>
                      <p className="text-xs text-orange-700">{item.label}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => updateQuantity(item.id, -1)}
                        className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-slate-700 transition hover:bg-slate-200"
                        aria-label="减少数量"
                      >
                        -
                      </button>
                      <span className="min-w-10 text-center text-base font-semibold">
                        {item.quantity}
                      </span>
                      <button
                        onClick={() => updateQuantity(item.id, 1)}
                        className="flex h-9 w-9 items-center justify-center rounded-lg bg-orange-500 text-white shadow-md shadow-orange-200/70 transition hover:bg-orange-600"
                        aria-label="增加数量"
                      >
                        +
                      </button>
                      <span className="ml-4 w-24 text-right text-lg font-bold text-slate-900">
                        ¥{(item.price * item.quantity).toFixed(2)}
                      </span>
                      <button
                        onClick={() => removeFromCart(item.id)}
                        className="ml-3 rounded-lg px-3 py-2 text-sm font-medium text-slate-500 transition hover:bg-slate-100 hover:text-red-600"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {cart.length > 0 && (
              <div className="mt-6 space-y-3 rounded-xl bg-slate-50 px-4 py-4 ring-1 ring-slate-100">
                <div className="flex items-center justify-between text-lg font-semibold text-slate-900">
                  <span>当前合计</span>
                  <span className="text-2xl text-orange-600">¥{calculateTotal().toFixed(2)}</span>
                </div>
                <button
                  onClick={handleCheckout}
                  className="w-full rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 py-3 text-base font-bold text-white shadow-lg shadow-orange-200/80 transition hover:scale-[1.01]"
                >
                  确认结账
                </button>
                <p className="text-xs text-slate-500">轻触屏幕即可完成收银，信息将实时同步。</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
