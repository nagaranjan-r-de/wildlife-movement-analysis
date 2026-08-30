import { useEffect, useState } from "react";
import { getDashboardSummary } from "../services/api";

import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import StatCard from "../components/StatCard";

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const result = await getDashboardSummary();
        setData(result);
      } catch (err) {
        console.error(err);
        setError("Unable to connect to the wildlife analysis backend.");
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, []);

  if (loading) {
    return (
      <div className="dashboard-state">
        <h2>Loading Wildlife Intelligence...</h2>
        <p>Connecting to the analysis engine.</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-state error">
        <h2>Backend Connection Error</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="app-layout">
      <Sidebar />

      <main className="main-content">
        <Header status={data.overall_conservation_status} />

        <section className="stats-grid">
          <StatCard
            icon="📊"
            title="Total Observations"
            value={data.total_observations.toLocaleString()}
            description="Wildlife tracking records"
          />

          <StatCard
            icon="🐾"
            title="Total Animals"
            value={data.total_animals.toLocaleString()}
            description="Tracked individuals"
          />

          <StatCard
            icon="🦌"
            title="Total Species"
            value={data.total_species.toLocaleString()}
            description="Species identified"
          />

          <StatCard
            icon="🗺️"
            title="Distance Tracked"
            value={`${data.total_distance_km.toLocaleString()} km`}
            description="Total movement distance"
          />

          <StatCard
            icon="🏃"
            title="Average Speed"
            value={`${data.average_speed_kmh} km/h`}
            description="Average movement speed"
          />

          <StatCard
            icon="⚠️"
            title="Average Risk"
            value={data.average_risk_score}
            description="Average conservation risk"
          />

          <StatCard
            icon="🚨"
            title="Movement Anomalies"
            value={data.movement_anomalies.toLocaleString()}
            description={`${data.anomaly_percentage}% of observations`}
          />

          <StatCard
            icon="🔴"
            title="High Risk"
            value={data.high_risk_observations.toLocaleString()}
            description={`${data.critical_risk_observations} critical observations`}
          />
        </section>

        <section className="risk-panel">
          <div className="risk-information">
            <p className="panel-label">CONSERVATION OVERVIEW</p>

            <h2>{data.overall_conservation_status}</h2>

            <p>
              Current wildlife movement analysis indicates elevated
              conservation risk across the monitored observations.
            </p>
          </div>

          <div className="risk-numbers">
            <div>
              <strong>{data.anomaly_percentage}%</strong>
              <span>Anomaly Rate</span>
            </div>

            <div>
              <strong>{data.maximum_risk_score}</strong>
              <span>Maximum Risk</span>
            </div>

            <div>
              <strong>{data.critical_risk_observations}</strong>
              <span>Critical Risk</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Dashboard;