"""
Flask REST API for Veri-Claim-AI.

Provides HTTP endpoints for claim-level
hallucination detection and verification.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.verification_service import (
    VerificationService,
    VerificationPipelineError
)


# ==================================================
# APPLICATION FACTORY
# ==================================================

def create_app(verification_service=None):
    """
    Create and configure the Flask application.

    Args:
        verification_service: Optional verification service.
            Allows mock services to be injected during testing.

    Returns:
        Configured Flask application.
    """

    app = Flask(__name__)

    # Enable Cross-Origin Resource Sharing.
    CORS(app)

    # Preserve dictionary insertion order in JSON responses.
    app.config["JSON_SORT_KEYS"] = False

    # Initialize the verification service.
    if verification_service is None:
        verification_service = VerificationService()

    # ==================================================
    # HEALTH CHECK ENDPOINT
    # ==================================================

    @app.route("/health", methods=["GET"])
    def health_check():
        """
        Check whether the backend is running.
        """

        return jsonify({
            "status": "success",
            "message": "Veri-Claim-AI backend is running.",
            "service": "Veri-Claim-AI"
        }), 200

    # ==================================================
    # TEXT VERIFICATION ENDPOINT
    # ==================================================

    @app.route("/api/verify", methods=["POST"])
    def verify_text():
        """
        Accept text and execute the verification pipeline.
        """

        # Validate request content type.
        if not request.is_json:

            return jsonify({
                "status": "error",
                "message": "Request must contain JSON data."
            }), 400

        # Parse JSON request.
        data = request.get_json(silent=True)

        if not isinstance(data, dict):

            return jsonify({
                "status": "error",
                "message": "Invalid JSON request body."
            }), 400

        # Extract input text.
        text = data.get("text")

        if text is None:

            return jsonify({
                "status": "error",
                "message": "Missing required field: text."
            }), 400

        # Execute verification pipeline.
        try:

            result = verification_service.verify(text)

            return jsonify({
                "status": "success",
                "data": result
            }), 200

        except VerificationPipelineError as error:

            return jsonify({
                "status": "error",
                "stage": error.stage,
                "message": error.message
            }), 422

        except Exception:

            app.logger.exception(
                "Unexpected verification API error."
            )

            return jsonify({
                "status": "error",
                "message": "An unexpected server error occurred."
            }), 500

    return app


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

app = create_app()


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )