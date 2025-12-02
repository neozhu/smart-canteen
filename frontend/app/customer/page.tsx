"use client";

import { useEffect, useState } from "react";

interface CartItem {
  id: string;
  label: string;
  name: string;
  price: number;
  quantity: number;
}

export default function CustomerScreen() {
  const [cart, setCart] = useState<CartItem[]>([]);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    // Listen for cart updates from operator screen
    if (typeof BroadcastChannel !== "undefined") {
      const channel = new BroadcastChannel("cart_sync");

      channel.onmessage = (event) => {
        setCart(event.data.cart || []);
        setTotal(event.data.total || 0);
      };

      return () => {
        channel.close();
      };
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-gray-100 text-slate-900">
      <div className="mx-auto max-w-4xl px-6 py-12 space-y-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-white/90 px-4 py-2 text-sm font-semibold text-orange-700 shadow-sm ring-1 ring-orange-100">
            <span className="h-2.5 w-2.5 rounded-full bg-orange-500 animate-pulse" />
            实时同步的顾客屏幕
          </div>
          <div className="space-y-1">
            <h1 className="text-4xl font-bold tracking-tight">欢迎光临 Smart Canteen</h1>
            <p className="text-gray-600">暖橙与柔和灰的清爽配色，让您的用餐确认更直观。</p>
          </div>
        </div>

        <div className="rounded-2xl bg-white/95 p-8 shadow-[0_28px_80px_-40px_rgba(17,24,39,0.4)] ring-1 ring-slate-100 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-orange-700">您的订单</p>
              <h2 className="text-2xl font-bold text-slate-900">自动识别的餐品明细</h2>
            </div>
            <div className="rounded-full bg-orange-50 px-4 py-2 text-sm font-semibold text-orange-700 ring-1 ring-orange-100">
              合计 ¥{total.toFixed(2)}
            </div>
          </div>

          {cart.length === 0 ? (
            <div className="flex flex-col items-center gap-3 rounded-xl bg-slate-50 px-4 py-10 text-center text-slate-500 ring-1 ring-slate-100">
              <span className="text-4xl" aria-hidden>提示</span>
              <p className="text-lg font-semibold text-slate-700">正在为您识别餐品...</p>
              <p className="text-sm text-slate-500">请稍候片刻，菜品将自动显示在此处。</p>
            </div>
          ) : (
            <div className="space-y-4">
              {cart.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center justify-between rounded-xl bg-gradient-to-r from-white to-orange-50/60 p-4 shadow-sm ring-1 ring-orange-100"
                >
                  <div>
                    <p className="text-xl font-semibold text-slate-900">{item.name}</p>
                    <p className="text-sm text-orange-700">数量: {item.quantity}</p>
                  </div>
                  <p className="text-2xl font-bold text-orange-600">
                    ¥{(item.price * item.quantity).toFixed(2)}
                  </p>
                </div>
              ))}

              <div className="space-y-3 rounded-xl bg-slate-50 px-4 py-5 ring-1 ring-slate-100">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-semibold text-slate-800">合计</span>
                  <span className="text-3xl font-bold text-orange-600">¥{total.toFixed(2)}</span>
                </div>
                <p className="text-sm text-slate-500">请确认无误后告知收银员完成结算。</p>
              </div>
            </div>
          )}
        </div>

        <div className="text-center text-sm text-slate-600">
          <p>请核对您的订单并告知收银员，感谢使用智能视觉结算系统。</p>
        </div>
      </div>
    </div>
  );
}
