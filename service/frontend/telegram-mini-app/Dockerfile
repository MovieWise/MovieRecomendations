FROM node:20-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ENV VITE_API_BASE_URL=/api/v1 \
    VITE_BACKEND_PROXY_TARGET=http://backend:8000

EXPOSE 5173

CMD ["npm", "run", "dev"]
