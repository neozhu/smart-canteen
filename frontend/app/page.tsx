"use client";

import { useEffect, useState } from "react";
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
    broadcastCart();
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
    broadcastCart();
  };

  const calculateTotal = () => {
    return cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
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
      broadcastCart();
    } catch (error) {
      console.error("Checkout failed:", error);
      alert("结账失败,请重试");
    }
  };

  const broadcastCart = () => {
    // Use BroadcastChannel to sync with customer screen
    if (typeof BroadcastChannel !== "undefined") {
      const channel = new BroadcastChannel("cart_sync");
      channel.postMessage({ cart, total: calculateTotal() });
      channel.close();
    }
  };

  useEffect(() => {
    broadcastCart();
  }, [cart]);

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">Smart Canteen - 操作员界面</h1>
          <a
            href="/annotate"
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition"
          >
            📝 数据标注
          </a>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Feed */}
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold mb-4">实时视频流</h2>
            <div className="relative bg-black rounded overflow-hidden mx-auto" style={{ aspectRatio: '4 / 3', maxWidth: '640px' }}>
              <img
                src={`${API_BASE}/video_feed`}
                alt="Camera Feed"
                className="w-full h-full object-contain"
              />
            </div>
            <div className="mt-4 text-sm text-gray-600">
              <p>检测状态: {isDetecting ? "识别中..." : "就绪"}</p>
              <p>
                购物车商品数:{" "}
                {cart.length}
              </p>
              <p className="text-xs text-gray-500 mt-1">自动识别间隔: 1秒</p>
            </div>
          </div>

          {/* Cart */}
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h2 className="text-xl font-semibold mb-4">购物车</h2>

            {cart.length === 0 ? (
              <p className="text-gray-500 text-center py-8">
                购物车为空,等待检测...
              </p>
            ) : (
              <div className="space-y-3">
                {cart.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between p-3 bg-gray-50 rounded"
                  >
                    <div className="flex-1">
                      <p className="font-medium">{item.name}</p>
                      <p className="text-sm text-gray-500">{item.label}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => updateQuantity(item.id, -1)}
                        className="w-8 h-8 rounded bg-gray-200 hover:bg-gray-300"
                      >
                        -
                      </button>
                      <span className="w-8 text-center font-medium">
                        {item.quantity}
                      </span>
                      <button
                        onClick={() => updateQuantity(item.id, 1)}
                        className="w-8 h-8 rounded bg-gray-200 hover:bg-gray-300"
                      >
                        +
                      </button>
                      <span className="ml-4 font-semibold w-20 text-right">
                        ¥{(item.price * item.quantity).toFixed(2)}
                      </span>
                      <button
                        onClick={() => removeFromCart(item.id)}
                        className="ml-2 text-red-500 hover:text-red-700"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {cart.length > 0 && (
              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex justify-between items-center text-2xl font-bold mb-4">
                  <span>总计:</span>
                  <span>¥{calculateTotal().toFixed(2)}</span>
                </div>
                <button
                  onClick={handleCheckout}
                  className="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-4 rounded-lg"
                >
                  确认结账
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
