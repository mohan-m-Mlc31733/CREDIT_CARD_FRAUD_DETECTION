/* =====================================================================
   FraudShield site configuration
   ---------------------------------------------------------------------
   API_BASE
     ""  (empty)  -> On GitHub Pages the scanner runs fully in the
                     browser (demo mode). When you run `python backend/app.py`
                     locally, the real Flask models are used automatically.
     "https://your-app.onrender.com"
                  -> Every /api/* call goes to your hosted Flask backend,
                     so the GitHub Pages site uses the real trained models.
                     (No trailing slash.)
   ===================================================================== */
window.FRAUDSHIELD_CONFIG = {
  API_BASE: "",
  API_TIMEOUT_MS: 25000
};
