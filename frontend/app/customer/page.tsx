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
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-white text-center mb-8">
          欢迎光临 Smart Canteen
        </h1>

        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <h2 className="text-2xl font-semibold mb-6 text-gray-800">
            您的订单
          </h2>

          {cart.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-3xl text-gray-400 mb-4">🍽️</p>
              <p className="text-xl text-gray-500">
                正在为您识别餐品...
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {cart.map((item) => (
                <div
                  key={item.id}
                  className="flex justify-between items-center p-4 bg-gray-50 rounded-lg"
                >
                  <div>
                    <p className="text-xl font-medium text-gray-800">
                      {item.name}
                    </p>
                    <p className="text-sm text-gray-500">
                      数量: {item.quantity}
                    </p>
                  </div>
                  <p className="text-2xl font-bold text-gray-800">
                    ¥{(item.price * item.quantity).toFixed(2)}
                  </p>
                </div>
              ))}

              <div className="pt-6 mt-6 border-t-2 border-gray-200">
                <div className="flex justify-between items-center">
                  <span className="text-3xl font-bold text-gray-800">
                    合计:
                  </span>
                  <span className="text-4xl font-bold text-green-600">
                    ¥{total.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="mt-8 text-center text-white text-lg">
          <p>请核对您的订单并告知收银员</p>
        </div>
      </div>
    </div>
  );
}
