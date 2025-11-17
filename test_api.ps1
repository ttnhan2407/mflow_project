# Danh sách input test
$inputs = @(
    @{water=90; light=85},
    @{water=70; light=75},  
    @{water=100; light=90},  
    @{water=50; light=60},   
    @{water=80; light=80},
    @{water=85; light=80}, 
    @{water=0; light=80},  
    @{water=100; light=80},
    @{water=0; light=90},
    @{water=1; light=15},
    @{water=1; light=82}
)

# URL API
$api_url = "http://127.0.0.1:8000/predict"

# Lặp qua từng input
foreach ($input in $inputs) {
    # Convert PowerShell object thành JSON
    $body = $input | ConvertTo-Json

    try {
        # Gửi POST request
        $response = Invoke-RestMethod -Uri $api_url -Method Post -Body $body -ContentType "application/json"
        
        # In kết quả
        $water = $input.water
        $light = $input.light
        $pred = if ($response.prediction -eq 1) {"Live"} else {"Die"}
        Write-Host "Water: $water, Light: $light => Prediction: $pred"
    }
    catch {
        Write-Host "Error: $($_.Exception.Message)"
    }
}
