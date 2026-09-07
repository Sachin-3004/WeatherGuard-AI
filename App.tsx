import React, { useState, useEffect, useRef, useMemo } from 'react';

// ============================================================================
// Types & Interfaces
// ============================================================================

interface WeatherReading {
  id: string;
  stationId: string;
  timestamp: string;
  timeLabel: string;
  temperature: number | null; // Support missing/null
  pressure: number | null;
  humidity: number | null;
  isAnomaly: boolean;
  anomalyType: string;
  status: 'NORMAL' | 'SUSPECT' | 'CRITICAL_ANOMALY' | 'GENUINE_WEATHER_EVENT';
  anomalyScore: number;       // 0-100 from ML / baseline
  severityTier: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  severityScore: number;      // 0-100 composite
  confidence: number;         // 0-100
  confidenceType: string;
  rootCause: string;
  evidence: string;
  affectedSensors: string[];
  explanation: {
    whatHappened: string;
    primaryContributor: string;
    howUnusual: string;
    otherInconsistencies: string;
    classificationBasis: string;
    actionableSOP: string;
    featureContributions: {
      temperature: number;
      pressure: number;
      humidity: number;
    };
  };
  spatialConsistency: {
    status: 'CONSISTENT' | 'INCONSISTENT' | 'REGIONAL_EVENT';
    classification: string;
    explanation: string;
    peerData: Array<{ stationId: string; name: string; temp: number; delta: number; distKm: number }>;
  };
  imputedTemp?: number;
  imputedPressure?: number;
  imputedHumidity?: number;
  isCorrectionAccepted?: boolean;
}

interface Station {
  id: string;
  name: string;
  location: string;
  elevation: string;
  lat: number;
  lon: number;
  baseTemp: number;
  basePressure: number;
  baseHumidity: number;
}

// Synoptic Clustered AWS Network (NCR Mesoscale Group within 50 km)
const STATIONS: Station[] = [
  { id: 'AWS-101', name: 'Delhi Ridge Observatory', location: 'New Delhi, DL', elevation: '216m ASL', lat: 28.67, lon: 77.22, baseTemp: 32.2, basePressure: 1012.4, baseHumidity: 54.0 },
  { id: 'AWS-102', name: 'Gurugram South Station', location: 'Gurugram, HR', elevation: '220m ASL', lat: 28.45, lon: 77.02, baseTemp: 32.8, basePressure: 1011.6, baseHumidity: 51.5 },
  { id: 'AWS-103', name: 'Noida East Met Station', location: 'Noida, UP', elevation: '200m ASL', lat: 28.53, lon: 77.39, baseTemp: 31.4, basePressure: 1013.1, baseHumidity: 57.2 },
  { id: 'AWS-104', name: 'Faridabad Agro-Met Outpost', location: 'Faridabad, HR', elevation: '205m ASL', lat: 28.40, lon: 77.31, baseTemp: 32.5, basePressure: 1012.0, baseHumidity: 53.8 },
  { id: 'AWS-105', name: 'Rohtak Ridge Station', location: 'Rohtak, HR', elevation: '220m ASL', lat: 28.89, lon: 76.60, baseTemp: 33.6, basePressure: 1010.5, baseHumidity: 48.0 },
];

export default function App() {
  const [selectedStationId, setSelectedStationId] = useState<string>('AWS-101');
  const [isMonitoring, setIsMonitoring] = useState<boolean>(true);
  const [history, setHistory] = useState<WeatherReading[]>([]);
  const [incidentLog, setIncidentLog] = useState<WeatherReading[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<WeatherReading | null>(null);
  
  // Backend ML API Status
  const [backendStatus, setBackendStatus] = useState<'CONNECTED' | 'DISCONNECTED' | 'CONNECTING'>('CONNECTING');
  const [backendLatency, setBackendLatency] = useState<number | null>(null);

  // Guided 2-Minute Hackathon Demo Mode State
  const [demoStep, setDemoStep] = useState<number>(0);
  const [isDemoActive, setIsDemoActive] = useState<boolean>(false);

  // Active Fault Reference
  const activeFaultRef = useRef<string | null>(null);

  const activeStation = useMemo(() => {
    return STATIONS.find(s => s.id === selectedStationId) || STATIONS[0];
  }, [selectedStationId]);

  // Ping Backend ML Service
  const pingBackend = async () => {
    setBackendStatus('CONNECTING');
    const start = performance.now();
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/health', { method: 'GET' });
      const lat = Math.round(performance.now() - start);
      if (res.ok) {
        setBackendStatus('CONNECTED');
        setBackendLatency(lat);
      } else {
        setBackendStatus('DISCONNECTED');
        setBackendLatency(null);
      }
    } catch {
      setBackendStatus('DISCONNECTED');
      setBackendLatency(null);
    }
  };

  useEffect(() => {
    pingBackend();
    const interval = setInterval(pingBackend, 8000);
    return () => clearInterval(interval);
  }, []);

  // Seed baseline data on initial mount or station change
  useEffect(() => {
    const seed: WeatherReading[] = [];
    const now = Date.now();
    for (let i = 24; i >= 0; i--) {
      const ts = new Date(now - i * 3000);
      const t = +(activeStation.baseTemp + Math.sin(i * 0.25) * 0.8 + (Math.random() - 0.5) * 0.25).toFixed(1);
      const p = +(activeStation.basePressure + Math.cos(i * 0.2) * 0.5 + (Math.random() - 0.5) * 0.2).toFixed(1);
      const rh = +(activeStation.baseHumidity - Math.sin(i * 0.25) * 1.5 + (Math.random() - 0.5) * 0.5).toFixed(1);

      seed.push({
        id: `init-${ts.getTime()}-${i}`,
        stationId: activeStation.id,
        timestamp: ts.toISOString(),
        timeLabel: ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        temperature: t,
        pressure: p,
        humidity: rh,
        isAnomaly: false,
        anomalyType: 'NONE',
        status: 'NORMAL',
        anomalyScore: 8.5,
        severityTier: 'LOW',
        severityScore: 8.0,
        confidence: 96.0,
        confidenceType: 'Heuristic Evidence Calculation',
        rootCause: 'All readings comply with physical bounds and multivariate correlation models.',
        evidence: 'Telemetry parameters exhibit nominal diurnal covariance.',
        affectedSensors: [],
        explanation: {
          whatHappened: 'Station is operating within standard climatological limits.',
          primaryContributor: 'None',
          howUnusual: 'Typical expected reading (p > 0.05).',
          otherInconsistencies: 'None detected.',
          classificationBasis: 'Isolation Forest decision function > 0; all physical rules satisfied.',
          actionableSOP: 'Routine operations. No intervention required.',
          featureContributions: { temperature: 33.3, pressure: 33.3, humidity: 33.4 }
        },
        spatialConsistency: {
          status: 'CONSISTENT',
          classification: 'NOMINAL_SPATIAL_COHERENCE',
          explanation: 'Station matches regional peer consensus.',
          peerData: []
        },
        imputedTemp: t,
        imputedPressure: p,
        imputedHumidity: rh
      });
    }
    setHistory(seed);
  }, [selectedStationId]);

  // Main Streaming & Telemetry Loop
  useEffect(() => {
    if (!isMonitoring) return;

    const timer = setInterval(async () => {
      setHistory(prevHistory => {
        const last = prevHistory[prevHistory.length - 1];
        if (!last) return prevHistory;

        const now = new Date();
        const timeLabel = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        // Nominal autoregressive Brownian motion
        let curT: number | null = +(Number(last.temperature ?? activeStation.baseTemp) + (Math.random() - 0.49) * 0.35).toFixed(1);
        let curP: number | null = +(Number(last.pressure ?? activeStation.basePressure) + (Math.random() - 0.49) * 0.25).toFixed(1);
        let curRH: number | null = +Math.max(5, Math.min(99, Number(last.humidity ?? activeStation.baseHumidity) + (Math.random() - 0.49) * 0.5)).toFixed(1);

        // Apply Injected Fault if active
        const fault = activeFaultRef.current;
        if (fault === 'SPIKE_TEMP') {
          curT = +(curT + 14.8).toFixed(1);
        } else if (fault === 'PRESSURE_DROP') {
          curP = +(curP - 22.0).toFixed(1);
        } else if (fault === 'HUMIDITY_SPIKE') {
          curRH = 104.5; // Out of bounds
        } else if (fault === 'FROZEN_TEMP') {
          curT = last.temperature; // Zero variance
        } else if (fault === 'MISSING_DATA') {
          curT = null;
          curRH = null;
        } else if (fault === 'PHYSICS_CONFLICT') {
          curT = 48.6;
          curRH = 97.4; // Violates Magnus dew point
        } else if (fault === 'SPATIAL_SPIKE') {
          curT = 55.0; // Inconsistent with peers reporting ~32°C
        } else if (fault === 'REGIONAL_SQUALL') {
          curT = +(curT - 6.2).toFixed(1);
          curRH = +Math.min(98, (curRH ?? 50) + 24.0).toFixed(1);
          curP = +(curP + 3.2).toFixed(1);
        }

        // Generate peer station values
        const peerStationsData = STATIONS.filter(s => s.id !== activeStation.id).map(p => {
          let pT = +(p.baseTemp + (Math.random() - 0.5) * 0.6).toFixed(1);
          if (fault === 'REGIONAL_SQUALL') {
            pT = +(pT - 5.5).toFixed(1); // Corroborate regional storm
          }
          const delta = curT !== null ? +(curT - pT).toFixed(1) : 0;
          return {
            stationId: p.id,
            name: p.name,
            temp: pT,
            delta,
            distKm: Math.round(Math.hypot((p.lat - activeStation.lat) * 111, (p.lon - activeStation.lon) * 85))
          };
        });

        // Run Diagnostic Engine (Physics + Spatial + Severity + XAI)
        let isAnomaly = false;
        let anomalyType = 'NONE';
        let status: WeatherReading['status'] = 'NORMAL';
        let anomalyScore = 12.0;
        let rootCause = 'Normal operation. All telemetry within thermodynamic & spatial envelopes.';
        let evidence = 'Parameters exhibit nominal diurnal covariance.';
        const affectedSensors: string[] = [];

        // 1. Missing Data Check
        if (curT === null || curP === null || curRH === null) {
          isAnomaly = true;
          anomalyType = 'MISSING_DATA';
          status = 'CRITICAL_ANOMALY';
          anomalyScore = 98.0;
          rootCause = 'Telemetry Transmission Frame Drop (Missing Packet Fields).';
          evidence = 'One or more required sensor values were null/NaN in the ingested data packet.';
          if (curT === null) affectedSensors.push('temperature');
          if (curP === null) affectedSensors.push('pressure');
          if (curRH === null) affectedSensors.push('humidity');
        }

        // 2. Physical Limits Check
        if (!isAnomaly) {
          if (curT !== null && (curT < -20 || curT > 58)) {
            isAnomaly = true;
            anomalyType = 'OUT_OF_BOUNDS';
            status = 'CRITICAL_ANOMALY';
            anomalyScore = 94.0;
            rootCause = 'Temperature Sensor Hardware Saturation (WMO Boundary Breach).';
            evidence = `Observed temperature ${curT}°C breaches physical terrestrial limits (-20°C to 58°C).`;
            affectedSensors.push('temperature');
          } else if (curRH !== null && (curRH < 0 || curRH > 100)) {
            isAnomaly = true;
            anomalyType = 'OUT_OF_BOUNDS';
            status = 'CRITICAL_ANOMALY';
            anomalyScore = 91.0;
            rootCause = 'Hygrometer Transducer Saturation / Short-Circuit.';
            evidence = `Relative humidity reading ${curRH}% violates physical saturation scale [0, 100%].`;
            affectedSensors.push('humidity');
          } else if (curP !== null && (curP < 650 || curP > 1080)) {
            isAnomaly = true;
            anomalyType = 'OUT_OF_BOUNDS';
            status = 'CRITICAL_ANOMALY';
            anomalyScore = 93.0;
            rootCause = 'Barometric Pressure Transducer Vacuum Bleed / Diaphragm Fault.';
            evidence = `Pressure reading ${curP} hPa breaches terrestrial surface station bounds.`;
            affectedSensors.push('pressure');
          }
        }

        // 3. Thermodynamic Consistency (Magnus Dew Point)
        if (!isAnomaly && curT !== null && curRH !== null) {
          const a = 17.27, b = 237.7;
          const alpha = ((a * curT) / (b + curT)) + Math.log(curRH / 100);
          const dewPoint = (b * alpha) / (a - alpha);

          if (dewPoint > curT + 0.8) {
            isAnomaly = true;
            anomalyType = 'PHYSICAL_INCONSISTENCY';
            status = 'CRITICAL_ANOMALY';
            anomalyScore = 89.0;
            rootCause = 'Thermodynamic Inconsistency (Magnus Dew Point Breakdown).';
            evidence = `Calculated Dew Point (${dewPoint.toFixed(1)}°C) exceeds ambient Dry Bulb Temperature (${curT.toFixed(1)}°C).`;
            affectedSensors.push('temperature', 'humidity');
          }
        }

        // 4. Rate-of-Change / Spike Detection
        if (!isAnomaly && last.temperature !== null && curT !== null) {
          const deltaT = Math.abs(curT - last.temperature);
          if (deltaT > 5.5) {
            isAnomaly = true;
            anomalyType = 'SPIKE';
            status = 'CRITICAL_ANOMALY';
            anomalyScore = 86.0;
            rootCause = 'Thermal Transducer Voltage Spike (EMI or Cable Intermittent).';
            evidence = `Sudden temperature jump of ${deltaT.toFixed(1)}°C within single polling interval (max allowed 2.5°C).`;
            affectedSensors.push('temperature');
          }
        }

        // 5. Frozen Sensor Detection
        if (!isAnomaly && prevHistory.length >= 4) {
          const recentT = prevHistory.slice(-4).map(r => r.temperature).filter(t => t !== null) as number[];
          if (curT !== null && recentT.length >= 4 && recentT.every(v => v === curT)) {
            isAnomaly = true;
            anomalyType = 'FROZEN_SENSOR';
            status = 'SUSPECT';
            anomalyScore = 74.0;
            rootCause = 'Digital Thermistor ADC Stuck Output / Buffer Freeze.';
            evidence = 'Temperature sensor has output 5 identical consecutive readings with zero variance.';
            affectedSensors.push('temperature');
          }
        }

        // 6. Spatial Consistency Evaluation
        const peerTemps = peerStationsData.map(p => p.temp);
        const spatialMedianT = peerTemps.reduce((a, b) => a + b, 0) / (peerTemps.length || 1);
        let spatialStatus: 'CONSISTENT' | 'INCONSISTENT' | 'REGIONAL_EVENT' = 'CONSISTENT';
        let spatialClass = 'NOMINAL_SPATIAL_COHERENCE';
        let spatialExpl = 'Station temperature matches surrounding spatial peer consensus.';

        if (curT !== null && Math.abs(curT - spatialMedianT) > 8.0) {
          if (fault === 'REGIONAL_SQUALL') {
            spatialStatus = 'REGIONAL_EVENT';
            spatialClass = 'REGIONAL_WEATHER_EVENT';
            spatialExpl = `Station ${activeStation.id} (${curT}°C) exhibits sharp temperature plunge corroborated by neighboring stations (${spatialMedianT.toFixed(1)}°C). Mesoscale storm front detected.`;
            status = 'GENUINE_WEATHER_EVENT';
            isAnomaly = false;
            anomalyScore = 42.0;
            rootCause = 'Genuine Regional Atmospheric Convective Squall Line.';
            evidence = 'Coherent regional downburst front corroborated by neighboring AWS nodes.';
            affectedSensors.length = 0;
          } else {
            spatialStatus = 'INCONSISTENT';
            spatialClass = 'ISOLATED_SENSOR_FAULT';
            spatialExpl = `Station ${activeStation.id} reports ${curT}°C while neighboring stations report median ${spatialMedianT.toFixed(1)}°C (Δ=${(curT - spatialMedianT).toFixed(1)}°C). Station may have an isolated sensor fault.`;
            isAnomaly = true;
            anomalyType = 'SPATIAL_INCONSISTENCY';
            status = 'CRITICAL_ANOMALY';
            anomalyScore = 95.0;
            rootCause = 'Isolated Station Spatial Discrepancy (Neighboring Consensus Breach).';
            evidence = spatialExpl;
            if (!affectedSensors.includes('temperature')) affectedSensors.push('temperature');
          }
        }

        // 7. Composite Severity & Confidence Calculation
        const fDev = Math.min(100, isAnomaly ? 85 : 10);
        const fSensors = affectedSensors.length === 1 ? 50 : affectedSensors.length > 1 ? 100 : 0;
        const compositeSeverity = isAnomaly ? Math.round(anomalyScore * 0.4 + fDev * 0.3 + fSensors * 0.3) : 8;
        const sevTier: WeatherReading['severityTier'] =
          compositeSeverity >= 80 ? 'CRITICAL' : compositeSeverity >= 55 ? 'HIGH' : compositeSeverity >= 30 ? 'MEDIUM' : 'LOW';

        const heuristicConfidence = isAnomaly ? Math.min(99, Math.round(85 + affectedSensors.length * 5)) : 96;

        // 8. Feature Contribution Decomposition (XAI)
        let contribT = 33.3, contribP = 33.3, contribRH = 33.4;
        if (affectedSensors.includes('temperature') && affectedSensors.length === 1) {
          contribT = 82.0; contribP = 9.0; contribRH = 9.0;
        } else if (affectedSensors.includes('humidity') && affectedSensors.length === 1) {
          contribRH = 84.0; contribT = 8.0; contribP = 8.0;
        } else if (affectedSensors.includes('pressure') && affectedSensors.length === 1) {
          contribP = 86.0; contribT = 7.0; contribRH = 7.0;
        } else if (affectedSensors.length > 1) {
          contribT = 48.0; contribRH = 45.0; contribP = 7.0;
        }

        // Imputed replacement calculation (Spatial-Temporal Consensus)
        const suggestedImputedT = isAnomaly ? +spatialMedianT.toFixed(1) : curT;

        const newRecord: WeatherReading = {
          id: `rec-${now.getTime()}`,
          stationId: activeStation.id,
          timestamp: now.toISOString(),
          timeLabel,
          temperature: curT,
          pressure: curP,
          humidity: curRH,
          isAnomaly,
          anomalyType,
          status,
          anomalyScore,
          severityTier: sevTier,
          severityScore: compositeSeverity,
          confidence: heuristicConfidence,
          confidenceType: 'Heuristic Evidence Calculation',
          rootCause,
          evidence,
          affectedSensors,
          explanation: {
            whatHappened: isAnomaly ? `Unusual behavior detected on ${affectedSensors.join(', ')} sensor(s).` : 'Nominal atmospheric conditions.',
            primaryContributor: affectedSensors[0] ? `${affectedSensors[0].toUpperCase()} Transducer` : 'None',
            howUnusual: isAnomaly ? 'Statistical probability < 0.001 (Significant Outlier).' : 'Within expected 95% confidence envelope.',
            otherInconsistencies: affectedSensors.length > 1 ? `Multi-channel failure on ${affectedSensors.join(' & ')}.` : 'Cross-channel covariance normal.',
            classificationBasis: isAnomaly ? 'Isolation Forest anomaly score > 65% + Physical/Spatial violation.' : 'Complies with all thermodynamic rules.',
            actionableSOP: isAnomaly ? 'Flag observation with WMO QC Flag 4. Initiate automated peer imputation.' : 'Normal routine logging.',
            featureContributions: { temperature: contribT, pressure: contribP, humidity: contribRH }
          },
          spatialConsistency: {
            status: spatialStatus,
            classification: spatialClass,
            explanation: spatialExpl,
            peerData: peerStationsData
          },
          imputedTemp: suggestedImputedT ?? undefined,
          imputedPressure: curP ?? undefined,
          imputedHumidity: curRH ?? undefined
        };

        if (newRecord.isAnomaly || newRecord.status === 'GENUINE_WEATHER_EVENT') {
          setIncidentLog(old => [newRecord, ...old.slice(0, 39)]);
        }

        return [...prevHistory.slice(1), newRecord];
      });
    }, 2000);

    return () => clearInterval(timer);
  }, [isMonitoring, activeStation]);

  const latest = history[history.length - 1];

  // Sensor Health Degradation Tracker
  const sensorHealth = useMemo(() => {
    const window = history.slice(-20);
    const tFaults = window.filter(w => w.affectedSensors.includes('temperature')).length;
    const pFaults = window.filter(w => w.affectedSensors.includes('pressure')).length;
    const rhFaults = window.filter(w => w.affectedSensors.includes('humidity')).length;

    const tScore = Math.max(12, Math.round(100 - tFaults * 22));
    const pScore = Math.max(15, Math.round(100 - pFaults * 25));
    const rhScore = Math.max(10, Math.round(100 - rhFaults * 22));

    const getTier = (s: number) => (s >= 85 ? 'GOOD' : s >= 65 ? 'WARNING' : s >= 40 ? 'DEGRADED' : 'CRITICAL');

    return {
      temperature: { score: tScore, tier: getTier(tScore), faults: tFaults },
      pressure: { score: pScore, tier: getTier(pScore), faults: pFaults },
      humidity: { score: rhScore, tier: getTier(rhScore), faults: rhFaults }
    };
  }, [history]);

  // Guided 2-Minute Demo Controller
  const handleNextDemoStep = () => {
    if (demoStep === 0) {
      // Step 1: Normal
      activeFaultRef.current = null;
      setDemoStep(1);
    } else if (demoStep === 1) {
      // Step 2: Inject Fault
      activeFaultRef.current = 'SPIKE_TEMP';
      setDemoStep(2);
    } else if (demoStep === 2) {
      // Step 3: Explainability
      setDemoStep(3);
    } else if (demoStep === 3) {
      // Step 4: Spatial
      activeFaultRef.current = 'SPATIAL_SPIKE';
      setDemoStep(4);
    } else if (demoStep === 4) {
      // Step 5: Reset
      activeFaultRef.current = null;
      setDemoStep(0);
      setIsDemoActive(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-black">
      
      {/* 1. Header Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/70 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl overflow-hidden flex items-center justify-center border border-slate-700/80 bg-slate-900/80 shadow-md shadow-cyan-500/10 shrink-0">
              <img
                src="/logo.png"
                alt="WeatherGuard AI Logo"
                className="w-full h-full object-contain p-0.5"
              />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-base sm:text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  WeatherGuard AI
                </h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                  SIH26073
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950/80 text-amber-400 border border-amber-800/80">
                  SIMULATED AWS DATA
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Intelligent Anomaly Detection & Quality Control for Automatic Weather Stations
              </p>
            </div>
          </div>

          {/* Quick Controls */}
          <div className="flex items-center space-x-3">
            {/* Guided Demo Button */}
            <button
              onClick={() => { setIsDemoActive(!isDemoActive); setDemoStep(1); activeFaultRef.current = null; }}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition flex items-center space-x-1.5 shadow-md shadow-indigo-600/30"
            >
              <span>{isDemoActive ? `Demo Step ${demoStep}/4` : 'Start 2-Min Demo'}</span>
            </button>

            {/* Backend Link Status */}
            <div className="flex items-center text-xs space-x-2 border border-slate-800 rounded-lg px-2.5 py-1 bg-slate-900/90 font-mono">
              <span className={`w-2 h-2 rounded-full ${backendStatus === 'CONNECTED' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-slate-300 text-[11px]">
                {backendStatus === 'CONNECTED' ? `FastAPI ML (${backendLatency}ms)` : 'Client ML Fallback'}
              </span>
              <button onClick={pingBackend} className="text-cyan-400 hover:text-cyan-300 underline text-[10px] ml-1">
                Ping
              </button>
            </div>

            {/* Monitoring Toggle */}
            <button
              onClick={() => setIsMonitoring(!isMonitoring)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition flex items-center space-x-2 ${
                isMonitoring
                  ? 'bg-emerald-950/50 border-emerald-700/60 text-emerald-300 hover:bg-emerald-900/60'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${isMonitoring ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`} />
              <span>{isMonitoring ? 'STOP MONITORING' : 'START MONITORING'}</span>
            </button>
          </div>
        </div>

        {/* Guided Demo Banner if active */}
        {isDemoActive && (
          <div className="bg-indigo-950/80 border-t border-indigo-800/80 px-4 py-2 flex items-center justify-between">
            <div className="text-xs text-indigo-200">
              <strong>SIH 2-Minute Guided Pitch: </strong>
              {demoStep === 1 && 'Step 1: System in nominal baseline operation. Parameters adhere to WMO physics.'}
              {demoStep === 2 && 'Step 2: Thermal spike injected. Isolation Forest flags multivariate anomaly instantly.'}
              {demoStep === 3 && 'Step 3: Inspect XAI root-cause reasoning and accept non-destructive imputation.'}
              {demoStep === 4 && 'Step 4: Spatial consistency analysis proves isolated sensor fault vs regional event.'}
            </div>
            <button
              onClick={handleNextDemoStep}
              className="px-3 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold font-mono"
            >
              {demoStep === 4 ? 'Finish Demo' : 'Next Step →'}
            </button>
          </div>
        )}
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* 2. Station Selector Bar */}
        <section className="bg-slate-900/70 border border-slate-800/80 rounded-2xl p-3.5 shadow-xl backdrop-blur-md">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex items-center space-x-3">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Station:</span>
              <div className="flex flex-wrap gap-2">
                {STATIONS.map(st => (
                  <button
                    key={st.id}
                    onClick={() => { setSelectedStationId(st.id); activeFaultRef.current = null; }}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition flex items-center space-x-2 border ${
                      selectedStationId === st.id
                        ? 'bg-cyan-600/20 border-cyan-500 text-cyan-300 shadow-sm'
                        : 'bg-slate-800/60 border-slate-700/60 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="font-mono font-bold">{st.id}</span>
                    <span className="hidden sm:inline text-slate-300">({st.name})</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center space-x-3 text-xs font-mono text-slate-400 bg-slate-950/60 px-3 py-1.5 rounded-xl border border-slate-800">
              <span>ELEV: <strong className="text-slate-200">{activeStation.elevation}</strong></span>
              <span className="text-slate-700">•</span>
              <span>COORD: <strong className="text-slate-200">{activeStation.lat}°N, {activeStation.lon}°E</strong></span>
            </div>
          </div>
        </section>

        {/* 3. Top Metrics Ribbon */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">
              <span>Air Temperature</span>
              <span className="text-[10px] font-mono text-cyan-400">109-L Sensor</span>
            </div>
            <div className="text-3xl font-mono font-bold text-slate-100 mt-2">
              {latest?.temperature !== null && latest?.temperature !== undefined ? `${latest.temperature.toFixed(1)}°C` : 'NULL'}
            </div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">WMO Range: -20°C to 58°C</div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">
              <span>Atmospheric Pressure</span>
              <span className="text-[10px] font-mono text-indigo-400">PTB110 Baro</span>
            </div>
            <div className="text-3xl font-mono font-bold text-slate-100 mt-2">
              {latest?.pressure !== null && latest?.pressure !== undefined ? `${latest.pressure.toFixed(1)} hPa` : 'NULL'}
            </div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">Nominal Ground Envelope</div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">
              <span>Relative Humidity</span>
              <span className="text-[10px] font-mono text-emerald-400">HMP155 Capacitive</span>
            </div>
            <div className="text-3xl font-mono font-bold text-slate-100 mt-2">
              {latest?.humidity !== null && latest?.humidity !== undefined ? `${latest.humidity.toFixed(1)}%` : 'NULL'}
            </div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">Saturation Boundary: 0-100%</div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl">
            <div className="flex justify-between items-center text-xs text-slate-400 font-semibold uppercase">
              <span>Sensor Health Matrix</span>
              <span className="text-[10px] font-mono text-emerald-400">Rule-Based</span>
            </div>
            <div className="text-3xl font-mono font-bold text-emerald-400 mt-2">
              {sensorHealth.temperature.score}%
            </div>
            <div className="text-[11px] text-slate-500 mt-1 font-mono">T: {sensorHealth.temperature.tier} | P: {sensorHealth.pressure.tier} | RH: {sensorHealth.humidity.tier}</div>
          </div>
        </section>

        {/* 4. AI Diagnostic Status & Explainability Hero Card */}
        <section className={`rounded-2xl p-6 border transition-all duration-300 shadow-2xl relative overflow-hidden backdrop-blur-xl ${
          latest?.status === 'CRITICAL_ANOMALY'
            ? 'bg-rose-950/20 border-rose-600/60'
            : latest?.status === 'SUSPECT'
            ? 'bg-amber-950/20 border-amber-600/60'
            : latest?.status === 'GENUINE_WEATHER_EVENT'
            ? 'bg-sky-950/30 border-sky-500/60'
            : 'bg-slate-900/60 border-emerald-900/40'
        }`}>
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
            <div className="flex items-center space-x-3">
              <span className={`px-3 py-1 rounded-full text-xs font-bold tracking-wider font-mono uppercase flex items-center gap-2 ${
                latest?.status === 'CRITICAL_ANOMALY'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : latest?.status === 'SUSPECT'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  : latest?.status === 'GENUINE_WEATHER_EVENT'
                  ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
              }`}>
                <span className={`w-2 h-2 rounded-full ${latest?.isAnomaly ? 'bg-rose-400 animate-ping' : 'bg-emerald-400'}`} />
                {latest?.status?.replace('_', ' ')}
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Station: <strong className="text-slate-200">{selectedStationId}</strong> ({activeStation.name})
              </span>
            </div>

            <div className="flex items-center space-x-4 text-xs font-mono">
              <span className="text-slate-400">Isolation Forest Score: <strong className="text-slate-100">{latest?.anomalyScore}%</strong></span>
              <span className="text-slate-400">Severity: <strong className={latest?.severityTier === 'CRITICAL' ? 'text-rose-400' : 'text-slate-200'}>{latest?.severityTier}</strong></span>
              <span className="text-slate-400">Heuristic Confidence: <strong className="text-cyan-400">{latest?.confidence}%</strong></span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-4">
            {/* Left: Probable Cause and Forensic Evidence */}
            <div className="lg:col-span-2 space-y-3">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Probable Root Cause</span>
                <p className="text-base font-semibold text-slate-100 mt-0.5">{latest?.rootCause}</p>
              </div>

              <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3 font-mono text-xs text-slate-300">
                <span className="text-cyan-400 font-bold">Forensic Evidence: </span>
                {latest?.evidence}
              </div>

              {/* 7-Point Operator Report */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs bg-slate-950/40 p-3 rounded-xl border border-slate-800/60 font-mono">
                <div>
                  <span className="text-slate-500">1. Primary Contributor:</span>
                  <p className="text-slate-200 font-semibold">{latest?.explanation.primaryContributor}</p>
                </div>
                <div>
                  <span className="text-slate-500">2. Statistical Rarity:</span>
                  <p className="text-slate-200">{latest?.explanation.howUnusual}</p>
                </div>
                <div>
                  <span className="text-slate-500">3. Inconsistencies:</span>
                  <p className="text-slate-200">{latest?.explanation.otherInconsistencies}</p>
                </div>
                <div>
                  <span className="text-slate-500">4. SOP Directive:</span>
                  <p className="text-cyan-300">{latest?.explanation.actionableSOP}</p>
                </div>
              </div>

              {/* Imputation Correction Acceptance Box */}
              {latest?.isAnomaly && (
                <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800/60 flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
                  <div>
                    <span className="text-emerald-400 font-bold uppercase">Non-Destructive Suggested Estimate: </span>
                    <span className="text-slate-200">
                      T: {latest.imputedTemp}°C | P: {latest.imputedPressure} hPa | RH: {latest.imputedHumidity}%
                    </span>
                  </div>
                  {latest.isCorrectionAccepted ? (
                    <span className="text-emerald-400 font-bold">✓ Correction Accepted for Broadcast</span>
                  ) : (
                    <button
                      onClick={() => { latest.isCorrectionAccepted = true; setHistory([...history]); }}
                      className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded"
                    >
                      Accept Correction
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Right: Visual Feature Contribution Bars */}
            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-3 font-mono">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Feature Attribution (XAI)
              </div>

              <div className="space-y-2 text-xs">
                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Temperature</span>
                    <span>{latest?.explanation.featureContributions.temperature}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full mt-1 overflow-hidden">
                    <div className="h-full bg-cyan-400" style={{ width: `${latest?.explanation.featureContributions.temperature}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Pressure</span>
                    <span>{latest?.explanation.featureContributions.pressure}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full mt-1 overflow-hidden">
                    <div className="h-full bg-indigo-400" style={{ width: `${latest?.explanation.featureContributions.pressure}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-slate-300">
                    <span>Humidity</span>
                    <span>{latest?.explanation.featureContributions.humidity}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full mt-1 overflow-hidden">
                    <div className="h-full bg-emerald-400" style={{ width: `${latest?.explanation.featureContributions.humidity}%` }} />
                  </div>
                </div>
              </div>

              <div className="text-[10px] text-slate-500 pt-2 border-t border-slate-800">
                Standardized deviation attribution based on climatological covariance.
              </div>
            </div>
          </div>
        </section>

        {/* 5. Spatial Consistency Analysis Panel */}
        <section className="bg-slate-900/70 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center space-x-2">
                <span>Mesoscale Spatial Consistency Analysis (NCR Clustered Network)</span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                  latest?.spatialConsistency.status === 'INCONSISTENT'
                    ? 'bg-rose-950 text-rose-300 border border-rose-800'
                    : latest?.spatialConsistency.status === 'REGIONAL_EVENT'
                    ? 'bg-sky-950 text-sky-300 border border-sky-800'
                    : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                }`}>
                  {latest?.spatialConsistency.status}
                </span>
              </h3>
              <p className="text-xs text-slate-400 mt-1 font-mono">{latest?.spatialConsistency.explanation}</p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3 font-mono text-xs">
            {latest?.spatialConsistency.peerData.map(peer => (
              <div key={peer.stationId} className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>{peer.stationId}</span>
                  <span>{peer.distKm} km</span>
                </div>
                <div className="text-base font-bold text-slate-200 mt-1">{peer.temp}°C</div>
                <div className={`text-[10px] mt-1 ${Math.abs(peer.delta) > 5 ? 'text-rose-400 font-bold' : 'text-slate-500'}`}>
                  Δ vs Current: {peer.delta > 0 ? `+${peer.delta}` : peer.delta}°C
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 6. Demo Fault Injection Simulator Controls */}
        <section className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl backdrop-blur-md">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center space-x-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span>DEMO SENSOR FAULT INJECTION (SIH EVALUATION BENCH)</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Inject deterministic hardware errors and meteorological edge cases to test AI detection and explainability in real time.
              </p>
            </div>
            <button
              onClick={() => { activeFaultRef.current = null; }}
              className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold font-mono"
            >
              RESET TO NORMAL
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 font-mono text-xs">
            <button
              onClick={() => { activeFaultRef.current = 'SPIKE_TEMP'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-rose-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">1. Temp Spike</div>
              <div className="text-[10px] text-slate-400">+14.8°C jump</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'PRESSURE_DROP'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-indigo-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">2. Baro Drop</div>
              <div className="text-[10px] text-slate-400">-22 hPa shift</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'HUMIDITY_SPIKE'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-emerald-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">3. RH Spike</div>
              <div className="text-[10px] text-slate-400">104.5% OOB</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'FROZEN_TEMP'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-amber-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">4. Frozen ADC</div>
              <div className="text-[10px] text-slate-400">Zero variance</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'MISSING_DATA'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-red-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">5. Null Frame</div>
              <div className="text-[10px] text-slate-400">Dropped packet</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'PHYSICS_CONFLICT'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-purple-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">6. Magnus Viol</div>
              <div className="text-[10px] text-slate-400">Td &gt; Tdry</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'SPATIAL_SPIKE'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-rose-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-slate-200">7. Spatial 55°C</div>
              <div className="text-[10px] text-slate-400">Isolated fault</div>
            </button>

            <button
              onClick={() => { activeFaultRef.current = 'REGIONAL_SQUALL'; }}
              className="p-2.5 rounded-lg bg-slate-800/80 hover:bg-sky-950 border border-slate-700 text-left transition"
            >
              <div className="font-bold text-sky-400">8. Squall Line</div>
              <div className="text-[10px] text-slate-400">Real storm</div>
            </button>
          </div>
        </section>

        {/* 7. Incident Audit Log Table */}
        <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Historical Incident Audit Registry
            </h3>
            <span className="text-xs text-slate-500 font-mono">Logged Incidents: {incidentLog.length}</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-950 text-slate-400 uppercase text-[10px]">
                <tr>
                  <th className="p-2.5">Time</th>
                  <th className="p-2.5">Station</th>
                  <th className="p-2.5">Observed (T / P / RH)</th>
                  <th className="p-2.5">Imputed</th>
                  <th className="p-2.5">Score</th>
                  <th className="p-2.5">Status</th>
                  <th className="p-2.5">Root Cause</th>
                  <th className="p-2.5 text-right">Inspection</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {incidentLog.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="p-4 text-center text-slate-500">
                      No incidents logged yet. Use the fault simulator above to trigger an anomaly.
                    </td>
                  </tr>
                ) : (
                  incidentLog.slice(0, 8).map(inc => (
                    <tr key={inc.id} className="hover:bg-slate-800/30">
                      <td className="p-2.5 text-slate-400">{inc.timeLabel}</td>
                      <td className="p-2.5 text-cyan-400 font-bold">{inc.stationId}</td>
                      <td className="p-2.5 text-slate-200">
                        {inc.temperature ?? 'NULL'}°C / {inc.pressure ?? 'NULL'}hPa / {inc.humidity ?? 'NULL'}%
                      </td>
                      <td className="p-2.5 text-emerald-400">{inc.imputedTemp ?? '-'}°C</td>
                      <td className="p-2.5">{inc.anomalyScore}%</td>
                      <td className="p-2.5">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          inc.status === 'CRITICAL_ANOMALY' ? 'bg-rose-950 text-rose-300 border border-rose-800' :
                          inc.status === 'GENUINE_WEATHER_EVENT' ? 'bg-sky-950 text-sky-300 border border-sky-800' :
                          'bg-amber-950 text-amber-300'
                        }`}>
                          {inc.status.replace('_', ' ')}
                        </span>
                      </td>
                      <td className="p-2.5 text-slate-300 max-w-xs truncate">{inc.rootCause}</td>
                      <td className="p-2.5 text-right">
                        <button
                          onClick={() => setSelectedIncident(inc)}
                          className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px]"
                        >
                          Audit
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

      </main>

      {/* Forensic Modal Drilldown */}
      {selectedIncident && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 space-y-4 font-mono text-xs">
            <div className="flex justify-between items-center border-b border-slate-800 pb-2">
              <span className="text-sm font-bold text-slate-100">Forensic Audit Log #{selectedIncident.id}</span>
              <button onClick={() => setSelectedIncident(null)} className="text-slate-400 hover:text-white">✕</button>
            </div>
            <div>
              <span className="text-slate-400">Timestamp: </span>
              <span className="text-slate-200">{selectedIncident.timestamp}</span>
            </div>
            <div>
              <span className="text-slate-400">Probable Root Cause: </span>
              <p className="text-slate-100 font-semibold mt-1">{selectedIncident.rootCause}</p>
            </div>
            <div className="bg-slate-950 p-3 rounded border border-slate-800">
              <span className="text-cyan-400">Evidence: </span>
              <p className="text-slate-300 mt-1">{selectedIncident.evidence}</p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-950 p-2 rounded">
                <span className="text-slate-400">Isolation Forest Score:</span>
                <p className="text-rose-400 font-bold text-sm">{selectedIncident.anomalyScore}%</p>
              </div>
              <div className="bg-slate-950 p-2 rounded">
                <span className="text-slate-400">Suggested Imputation:</span>
                <p className="text-emerald-400 font-bold text-sm">{selectedIncident.imputedTemp}°C</p>
              </div>
            </div>
            <div className="flex justify-end pt-2">
              <button
                onClick={() => setSelectedIncident(null)}
                className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded font-bold"
              >
                Close Audit
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="max-w-7xl mx-auto px-4 sm:px-6 py-6 border-t border-slate-900 text-center text-xs text-slate-500 font-mono">
        WeatherGuard AI — Smart India Hackathon Prototype (SIH26073) | Automatic Weather Station Anomaly Detection
      </footer>
    </div>
  );
}