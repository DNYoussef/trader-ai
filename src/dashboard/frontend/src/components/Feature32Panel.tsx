import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

interface Feature32PanelProps {
  features?: {
    values: number[];
    names: string[];
    timestamp: string;
    categories?: {
      market: number[];
      inequality: number[];
      risk: number[];
      ai: number[];
    };
    metadata?: {
      source: string;
      data_quality: string;
    };
  };
  isLoading?: boolean;
}

// Feature visualization component in 3D
const FeatureSphere: React.FC<{
  position: [number, number, number];
  value: number;
  name: string;
  index: number;
  category: string;
}> = ({ position, value, name, index, category }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Normalize value to 0-1 range
  const normalizedValue = Math.abs(value) / (Math.abs(value) + 1);

  // Color based on category
  const categoryColors = {
    market: '#3B82F6',      // Blue
    inequality: '#F59E0B',  // Orange
    risk: '#EF4444',        // Red
    ai: '#10B981'          // Green
  };

  const color = categoryColors[category as keyof typeof categoryColors] || '#6B7280';

  // Animate rotation
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.001 * (index + 1);
      meshRef.current.rotation.y += 0.002 * (index + 1);

      // Pulse effect based on value
      const scale = 0.3 + normalizedValue * 0.4 + (hovered ? 0.2 : 0);
      meshRef.current.scale.setScalar(scale);
    }
  });

  return (
    <group position={position}>
      <Sphere
        ref={meshRef}
        args={[0.3, 16, 16]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={normalizedValue * 0.5}
          metalness={0.8}
          roughness={0.2}
          opacity={0.8 + normalizedValue * 0.2}
          transparent
        />
      </Sphere>

      {/* Feature label */}
      <Text
        position={[0, -0.6, 0]}
        fontSize={0.15}
        color={hovered ? '#FFFFFF' : '#9CA3AF'}
        anchorX="center"
        anchorY="middle"
      >
        {`F${index + 1}`}
      </Text>

      {/* Value indicator */}
      {hovered && (
        <Text
          position={[0, 0.6, 0]}
          fontSize={0.12}
          color="#FFFFFF"
          anchorX="center"
          anchorY="middle"
        >
          {value.toFixed(3)}
        </Text>
      )}
    </group>
  );
};

// Connection lines between related features
const FeatureConnections: React.FC<{
  features: number[];
  positions: [number, number, number][];
}> = ({ features, positions }) => {
  const connections = useMemo(() => {
    const lines: Array<[[number, number, number], [number, number, number]]> = [];

    // Connect features with high correlation (simplified)
    for (let i = 0; i < features.length - 1; i++) {
      for (let j = i + 1; j < Math.min(i + 3, features.length); j++) {
        const correlation = Math.abs(features[i] - features[j]) < 0.2 ? 1 : 0;
        if (correlation > 0.5) {
          lines.push([positions[i], positions[j]]);
        }
      }
    }

    return lines;
  }, [features, positions]);

  return (
    <>
      {connections.map((points, idx) => (
        <Line
          key={idx}
          points={points}
          color="#4B5563"
          lineWidth={0.5}
          opacity={0.3}
          transparent
        />
      ))}
    </>
  );
};

// Main 3D scene component
const Feature3DVisualization: React.FC<{
  features: number[];
  names: string[];
  categories: any;
}> = ({ features, names, categories }) => {
  // Calculate positions for features in a spiral pattern
  const positions = useMemo(() => {
    const result: [number, number, number][] = [];
    const numFeatures = features.length;

    for (let i = 0; i < numFeatures; i++) {
      const angle = (i / numFeatures) * Math.PI * 4; // Two full rotations
      const radius = 3 + i * 0.1;
      const height = (i / numFeatures) * 3 - 1.5;

      result.push([
        Math.cos(angle) * radius,
        height,
        Math.sin(angle) * radius
      ]);
    }

    return result;
  }, [features]);

  // Determine category for each feature
  const getCategory = (index: number) => {
    if (categories?.market?.includes(index)) return 'market';
    if (categories?.inequality?.includes(index)) return 'inequality';
    if (categories?.risk?.includes(index)) return 'risk';
    if (categories?.ai?.includes(index)) return 'ai';
    return 'unknown';
  };

  return (
    <>
      {/* Ambient and directional lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#FF0099" />

      {/* Feature spheres */}
      {features.map((value, idx) => (
        <FeatureSphere
          key={idx}
          position={positions[idx]}
          value={value}
          name={names[idx] || `Feature ${idx + 1}`}
          index={idx}
          category={getCategory(idx)}
        />
      ))}

      {/* Connection lines */}
      <FeatureConnections features={features} positions={positions} />

      {/* Center reference sphere */}
      <Sphere args={[0.2, 32, 32]} position={[0, 0, 0]}>
        <meshStandardMaterial
          color="#FFFFFF"
          emissive="#FFFFFF"
          emissiveIntensity={0.2}
        />
      </Sphere>

      {/* Category labels */}
      <Text position={[5, 2, 0]} fontSize={0.3} color="#3B82F6">Market</Text>
      <Text position={[-5, 2, 0]} fontSize={0.3} color="#F59E0B">Inequality</Text>
      <Text position={[0, 2, 5]} fontSize={0.3} color="#EF4444">Risk</Text>
      <Text position={[0, 2, -5]} fontSize={0.3} color="#10B981">AI</Text>
    </>
  );
};

// Feature list panel for detailed view
const FeatureListPanel: React.FC<{
  features: number[];
  names: string[];
  categories: any;
}> = ({ features, names, categories }) => {
  const categoryConfig = {
    market: { name: 'Market Indicators', color: 'bg-blue-500' },
    inequality: { name: 'Inequality Metrics', color: 'bg-orange-500' },
    risk: { name: 'Risk Signals', color: 'bg-red-500' },
    ai: { name: 'AI Predictions', color: 'bg-green-500' }
  };

  const getCategory = (index: number) => {
    if (categories?.market?.includes(index)) return 'market';
    if (categories?.inequality?.includes(index)) return 'inequality';
    if (categories?.risk?.includes(index)) return 'risk';
    if (categories?.ai?.includes(index)) return 'ai';
    return 'unknown';
  };

  const getValueColor = (value: number) => {
    const normalized = Math.abs(value);
    if (normalized > 1) return 'text-red-600';
    if (normalized > 0.5) return 'text-yellow-600';
    return 'text-green-600';
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4">
      {Object.entries(categoryConfig).map(([key, config]) => (
        <div key={key} className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
          <div className="flex items-center mb-3">
            <div className={`w-3 h-3 rounded-full ${config.color} mr-2`}></div>
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
              {config.name}
            </h3>
          </div>
          <div className="space-y-2">
            {features.map((value, idx) => {
              if (getCategory(idx) !== key) return null;

              return (
                <div key={idx} className="flex justify-between items-center">
                  <span className="text-xs text-gray-600 dark:text-gray-400">
                    {names[idx] || `Feature ${idx + 1}`}
                  </span>
                  <span className={`text-xs font-mono ${getValueColor(value)}`}>
                    {value.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
};

// Main component
export const Feature32Panel: React.FC<Feature32PanelProps> = ({
  features,
  isLoading = false
}) => {
  const [viewMode, setViewMode] = useState<'3d' | 'list'>('3d');
  const [autoRotate, setAutoRotate] = useState(true);

  // Fetch features if not provided
  useEffect(() => {
    if (!features) {
      // Fetch from API
      fetch('/api/features/realtime')
        .then(res => res.json())
        .then(data => {
          console.log('Fetched features:', data);
        })
        .catch(err => console.error('Error fetching features:', err));
    }
  }, [features]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading 32 features...</p>
        </div>
      </div>
    );
  }

  if (!features || !features.values || features.values.length === 0) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 dark:bg-gray-900 rounded-lg">
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400">No feature data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              AI Model Input Features (32D)
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Real-time data from {features.metadata?.source || 'unknown'} •
              Quality: {features.metadata?.data_quality || 'unknown'}
            </p>
          </div>

          <div className="flex items-center space-x-4">
            {/* View mode toggle */}
            <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
              <button
                onClick={() => setViewMode('3d')}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  viewMode === '3d'
                    ? 'bg-white dark:bg-gray-600 text-indigo-600 dark:text-indigo-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                3D View
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  viewMode === 'list'
                    ? 'bg-white dark:bg-gray-600 text-indigo-600 dark:text-indigo-400'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
              >
                List View
              </button>
            </div>

            {/* Auto-rotate toggle (for 3D view) */}
            {viewMode === '3d' && (
              <button
                onClick={() => setAutoRotate(!autoRotate)}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  autoRotate
                    ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300'
                    : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                }`}
              >
                {autoRotate ? 'Auto-Rotate ON' : 'Auto-Rotate OFF'}
              </button>
            )}

            {/* Timestamp */}
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {new Date(features.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>

      {/* Content */}
      {viewMode === '3d' ? (
        <div className="h-[600px] bg-gray-900">
          <Canvas camera={{ position: [10, 5, 10], fov: 50 }}>
            <Feature3DVisualization
              features={features.values}
              names={features.names}
              categories={features.categories}
            />
            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              autoRotate={autoRotate}
              autoRotateSpeed={0.5}
            />

            {/* Grid helper */}
            <gridHelper args={[20, 20, '#4B5563', '#374151']} />
          </Canvas>
        </div>
      ) : (
        <FeatureListPanel
          features={features.values}
          names={features.names}
          categories={features.categories}
        />
      )}

      {/* Statistics footer */}
      <div className="px-6 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
        <div className="flex justify-between text-sm">
          <div className="flex space-x-6">
            <span className="text-gray-600 dark:text-gray-400">
              Min: <span className="font-mono text-gray-900 dark:text-white">
                {Math.min(...features.values).toFixed(3)}
              </span>
            </span>
            <span className="text-gray-600 dark:text-gray-400">
              Max: <span className="font-mono text-gray-900 dark:text-white">
                {Math.max(...features.values).toFixed(3)}
              </span>
            </span>
            <span className="text-gray-600 dark:text-gray-400">
              Avg: <span className="font-mono text-gray-900 dark:text-white">
                {(features.values.reduce((a, b) => a + b, 0) / features.values.length).toFixed(3)}
              </span>
            </span>
          </div>

          <div className="flex space-x-4">
            <span className="flex items-center">
              <div className="w-2 h-2 bg-blue-500 rounded-full mr-1"></div>
              <span className="text-gray-600 dark:text-gray-400">Market</span>
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-orange-500 rounded-full mr-1"></div>
              <span className="text-gray-600 dark:text-gray-400">Inequality</span>
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-red-500 rounded-full mr-1"></div>
              <span className="text-gray-600 dark:text-gray-400">Risk</span>
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
              <span className="text-gray-600 dark:text-gray-400">AI</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Feature32Panel;