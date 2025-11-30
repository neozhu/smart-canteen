# Smart-Canteen Frontend

🖥️ Next.js 14 web application for the Smart-Canteen AI vision checkout system. Provides operator interface, customer display, and data annotation tools.

## 🌟 Features

- **Operator Interface** (`/`): Real-time detection, shopping cart management, video stream
- **Customer Display** (`/customer`): Synchronized cart view with pricing
- **Annotation Tool** (`/annotate`): Data collection and labeling with YOLOv8n assistance
- **On-Demand Detection**: Frontend-triggered inference (1-second intervals)
- **Cart Replacement Mode**: Detection results replace cart entirely (not accumulative)
- **Live Video Stream**: MJPEG stream from backend with bounding box overlay

## 🚀 Quick Start

### Installation

```bash
npm install
# or
yarn install
```

### Development Server

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) to view the operator interface.

### Build for Production

```bash
npm run build
npm start
```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── page.tsx              # Operator interface (main detection page)
│   ├── customer/
│   │   └── page.tsx          # Customer-facing display
│   ├── annotate/
│   │   └── page.tsx          # Data annotation tool
│   ├── layout.tsx            # Root layout (metadata, fonts)
│   └── globals.css           # Global styles (TailwindCSS)
├── public/                   # Static assets
├── package.json              # Dependencies and scripts
└── next.config.js            # Next.js configuration
```

## 🎯 Page Descriptions

### Operator Interface (`/`)

Main checkout interface for operators:

**Features:**
- Live MJPEG video stream from backend (`http://localhost:8000/api/video_feed`)
- On-demand detection trigger (every 1 second)
- Real-time shopping cart with item grouping
- Detection status indicator ("🔍 Detecting..." / "✓ Ready")
- Manual cart clear button
- Auto-scrolling cart list

**Key Components:**
- Video feed display (640x480)
- Detection polling logic (`useEffect` with 1s throttle)
- Cart state management (replacement mode)
- Item price calculation and total sum

**API Calls:**
```typescript
// Trigger detection (every 1 second)
GET http://localhost:8000/api/detect_once

// Response:
{
  "detections": [
    {"label": "coin", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

**Cart Logic:**
```typescript
// Replacement mode (NOT accumulative)
if (detections.length === 0) {
  setCart([]); // Clear cart if nothing detected
} else {
  const newCart = /* group by label */;
  setCart(newCart); // Replace entirely
}
```

### Customer Display (`/customer`)

Customer-facing screen for price transparency:

**Features:**
- Synchronized with operator cart
- Large, readable fonts
- Item-by-item breakdown
- Total amount prominently displayed
- Auto-updates every 200ms

**Data Sync:**
```typescript
// Poll current detection state
GET http://localhost:8000/api/current_detection
```

### Annotation Tool (`/annotate`)

Data collection interface for training:

**Features:**
- Live camera preview
- Label dropdown (reads from backend `classes.json`)
- One-click capture with annotation
- Training progress monitoring
- Model training trigger
- YOLOv8n pre-trained model assistance (automatic bbox suggestions)

**Workflow:**
1. Select object label from dropdown
2. Place object in camera view
3. Click "📸 Capture" (bbox auto-detected by YOLOv8n)
4. Repeat 20-30 times per class
5. Click "🚀 Start Training"
6. Monitor training progress (150 epochs, ~30-45 min)

**API Calls:**
```typescript
// Save annotation
POST http://localhost:8000/api/save_annotation
{
  "image": "base64_encoded_image",
  "label": "coin",
  "bbox": [x1, y1, x2, y2]
}

// Start training
POST http://localhost:8000/api/start_training

// Check progress
GET http://localhost:8000/api/training_status
```

## 🛠️ Configuration

### Backend API URL

Default: `http://localhost:8000`

To change, update API calls in:
- `app/page.tsx` (operator interface)
- `app/customer/page.tsx` (customer display)
- `app/annotate/page.tsx` (annotation tool)

### Detection Interval

Default: 1000ms (1 second)

Modify in `app/page.tsx`:
```typescript
const now = Date.now();
if (!isDetecting && now - lastDetectionTime > 1000) { // Change this value
  detectOnce();
}
```

### Video Stream URL

Default: `http://localhost:8000/api/video_feed`

Update `<img>` src in `app/page.tsx`:
```typescript
<img
  src="http://localhost:8000/api/video_feed"
  alt="Camera Feed"
  className="w-full h-auto"
/>
```

## 🎨 Styling

Built with **TailwindCSS 3.x**:

**Key Classes:**
- `bg-gradient-to-br from-blue-50 to-purple-50` - Background gradients
- `shadow-2xl` - Card shadows
- `hover:scale-105` - Interactive hover effects
- `animate-pulse` - Detection status indicator

**Custom Styles:**
Edit `app/globals.css` for global overrides.

## 📦 Dependencies

### Core
- `next`: 14.0.0 - React framework
- `react`: 18.0.0 - UI library
- `react-dom`: 18.0.0 - React DOM renderer

### HTTP Client
- `axios`: ^1.6.0 - API requests

### Styling
- `tailwindcss`: 3.x - Utility-first CSS
- `@tailwindcss/forms`: Plugin for form styling

### TypeScript
- `typescript`: 5.x
- `@types/node`: Type definitions
- `@types/react`: Type definitions

## 🔄 State Management

Using React hooks (no external state library):

**Operator Interface State:**
```typescript
const [cart, setCart] = useState<CartItem[]>([]);
const [isDetecting, setIsDetecting] = useState(false);
const [lastDetectionTime, setLastDetectionTime] = useState(0);
```

**Cart Item Interface:**
```typescript
interface CartItem {
  id: string;           // Unique identifier (label_timestamp)
  label: string;        // Object class label
  name: string;         // Display name (from price_map)
  price: number;        // Unit price
  quantity: number;     // Count (grouped by label)
}
```

## 🐛 Troubleshooting

### API Connection Issues

**Problem**: "Failed to fetch" errors

**Solutions:**
1. Verify backend is running: `curl http://localhost:8000/api/camera_status`
2. Check CORS settings in backend `main.py`
3. Ensure no firewall blocking localhost:8000

### Cart Not Updating

**Problem**: Cart doesn't reflect detections

**Debug:**
1. Open browser console (F12)
2. Check for API errors in Network tab
3. Verify detection API returns data: `curl http://localhost:8000/api/detect_once`
4. Check `isDetecting` flag (should toggle every 1s)

### Video Stream Not Loading

**Problem**: Black screen or broken image

**Solutions:**
1. Verify MJPEG endpoint: Open `http://localhost:8000/api/video_feed` in browser
2. Check backend camera initialization
3. Ensure camera not in use by other applications
4. Check browser console for CORS errors

### Build Errors

**Problem**: TypeScript compilation errors

**Solutions:**
```bash
# Clear cache and rebuild
rm -rf .next
npm run build

# Check TypeScript version compatibility
npm list typescript
```

## 🚀 Deployment

### Standalone Build

```bash
npm run build
npm start
```

Runs on port 3000 (configure in `package.json`).

### Docker (Optional)

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Vercel (Not Recommended)

This app requires local backend connection. For Vercel deployment:
1. Modify API URLs to point to public backend
2. Handle CORS properly
3. Consider latency for video streaming

## 📚 Learn More

### Next.js Resources
- [Next.js Documentation](https://nextjs.org/docs) - Features and API
- [Learn Next.js](https://nextjs.org/learn) - Interactive tutorial
- [Next.js GitHub](https://github.com/vercel/next.js) - Feedback and contributions

### Project-Specific
- See root [README.md](../README.md) for full system documentation
- Backend API documentation: Check `backend/main.py` for endpoint details
- Training pipeline: See `backend/annotation.py`

## 🤝 Contributing

When contributing to frontend:
1. Follow existing code style (TypeScript + Tailwind)
2. Test all API integrations with backend
3. Ensure responsive design (mobile/tablet/desktop)
4. Update this README for new features

## 📄 License

MIT License - see root LICENSE file

---

**Built with Next.js 14** • **Powered by YOLOv8** • **Styled with Tailwind**
