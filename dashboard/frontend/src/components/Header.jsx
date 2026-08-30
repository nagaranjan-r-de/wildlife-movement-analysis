function Header({ status }) {
  return (
    <header className="top-header">
      <div>
        <p className="header-eyebrow">
          WILDLIFE CONSERVATION INTELLIGENCE
        </p>

        <h1>Wildlife Movement Dashboard</h1>

        <p className="header-subtitle">
          Monitor movement patterns, behavioral changes, anomalies,
          and conservation risk.
        </p>
      </div>

      <div className="conservation-status">
        <span className="risk-dot"></span>

        <div>
          <small>CONSERVATION STATUS</small>
          <strong>{status || "Loading..."}</strong>
        </div>
      </div>
    </header>
  );
}

export default Header;
