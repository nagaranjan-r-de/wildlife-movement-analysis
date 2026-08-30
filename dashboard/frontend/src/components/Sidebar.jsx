import {
  LayoutDashboard,
  Map,
  PawPrint,
  Brain,
  TriangleAlert,
  ShieldCheck,
  CloudSun,
} from "lucide-react";

const menuItems = [
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Movement", icon: Map },
  { label: "Species", icon: PawPrint },
  { label: "Behavior", icon: Brain },
  { label: "Anomalies", icon: TriangleAlert },
  { label: "Conservation", icon: ShieldCheck },
  { label: "Environment", icon: CloudSun },
];

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon">🐾</div>

        <div>
          <h2>WildTrack</h2>
          <span>Conservation Intelligence</span>
        </div>
      </div>

      <nav className="sidebar-nav">
        <p className="nav-title">ANALYTICS</p>

        {menuItems.map((item, index) => {
          const Icon = item.icon;

          return (
            <button
              key={item.label}
              className={`nav-item ${index === 0 ? "active" : ""}`}
            >
              <Icon size={19} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="system-status">
          <span className="status-online"></span>

          <div>
            <strong>System Online</strong>
            <small>Analysis engine active</small>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
