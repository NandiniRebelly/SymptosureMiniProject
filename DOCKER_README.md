# Docker Deployment Guide

This guide explains how to deploy the Symptoms Checker API using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 2GB RAM available for containers

## Quick Start

### 1. Basic Deployment

```bash
# Build and start the API service
docker-compose up --build

# Run in background
docker-compose up -d --build
```

The API will be available at `http://localhost:8000`

### 2. Production Deployment (with Nginx)

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d --build
```

The API will be available at `http://localhost:80`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud credentials | `/secrets/google.json` |
| `PYTHONPATH` | Python path | `/app` |
| `PYTHONUNBUFFERED` | Python output buffering | `1` |

### Volume Mounts

- `./artifacts:/app/artifacts:ro` - Model artifacts (read-only)
- `./secrets:/secrets:ro` - Google Cloud credentials (read-only)

## Google Cloud Setup (Optional)

### 1. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create symptoms-checker

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:symptoms-checker@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/speech.client"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:symptoms-checker@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/texttospeech.client"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:symptoms-checker@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/translate.client"
```

### 2. Download Credentials

```bash
# Create secrets directory
mkdir -p secrets

# Download service account key
gcloud iam service-accounts keys create secrets/google.json \
    --iam-account=symptoms-checker@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 3. Deploy with Credentials

```bash
docker-compose up -d --build
```

## Health Checks

The container includes health checks:

```bash
# Check container health
docker-compose ps

# View health check logs
docker-compose logs symptom-api
```

## Scaling

### Horizontal Scaling

```bash
# Scale to 3 instances
docker-compose up -d --scale symptom-api=3
```

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  symptom-api:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
```

## Monitoring

### Logs

```bash
# View logs
docker-compose logs -f symptom-api

# View last 100 lines
docker-compose logs --tail=100 symptom-api
```

### Metrics

The API exposes metrics at `/health`:

```bash
# Check API health
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "artifacts_loaded": true
}
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Artifacts Not Found**
   ```bash
   # Ensure artifacts directory exists
   ls -la artifacts/
   
   # Check volume mount
   docker-compose exec symptom-api ls -la /app/artifacts
   ```

3. **Google Cloud Authentication**
   ```bash
   # Check credentials file
   ls -la secrets/google.json
   
   # Test inside container
   docker-compose exec symptom-api env | grep GOOGLE
   ```

4. **Memory Issues**
   ```bash
   # Check container memory usage
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

### Debug Mode

```bash
# Run with debug logging
docker-compose up --build

# Access container shell
docker-compose exec symptom-api bash

# Check Python environment
docker-compose exec symptom-api python -c "import sys; print(sys.path)"
```

## Security Considerations

### Production Checklist

- [ ] Use HTTPS in production
- [ ] Set up proper firewall rules
- [ ] Use secrets management (Docker Secrets, Kubernetes Secrets)
- [ ] Regular security updates
- [ ] Monitor resource usage
- [ ] Set up log aggregation
- [ ] Configure backup strategies

### Security Headers

The nginx configuration includes security headers:

- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`

## Performance Tuning

### Worker Processes

The API runs with 4 workers by default. Adjust based on CPU cores:

```dockerfile
# In Dockerfile
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]
```

### Memory Optimization

```yaml
# In docker-compose.yml
services:
  symptom-api:
    environment:
      - PYTHONOPTIMIZE=1
      - PYTHONDONTWRITEBYTECODE=1
```

## Backup and Recovery

### Backup Artifacts

```bash
# Backup artifacts
tar -czf artifacts-backup-$(date +%Y%m%d).tar.gz artifacts/

# Restore artifacts
tar -xzf artifacts-backup-20240101.tar.gz
```

### Container Backup

```bash
# Save container image
docker save symptoms-checker-api > symptoms-checker.tar

# Load container image
docker load < symptoms-checker.tar
```

## Updates

### Rolling Updates

```bash
# Update application
docker-compose pull
docker-compose up -d --build

# Zero-downtime update
docker-compose up -d --no-deps symptom-api
```

### Version Management

```bash
# Tag specific version
docker tag symptoms-checker-api:latest symptoms-checker-api:v1.0.0

# Deploy specific version
docker-compose up -d --build
```

## Support

For issues and questions:

1. Check container logs: `docker-compose logs symptom-api`
2. Verify health status: `curl http://localhost:8000/health`
3. Test API endpoint: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"input":"test","input_type":"text","language":"en","mode":"text"}'`
4. Review this documentation
5. Check GitHub issues for known problems
