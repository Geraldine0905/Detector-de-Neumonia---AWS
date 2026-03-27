# PneumoScan — VM de despliegue EKS
# Herramientas: AWS CLI v2, kubectl, eksctl, Docker
# Uso:
#   vagrant up      → crea y provisiona la VM (~5-10 min)
#   vagrant ssh     → entra a la VM
#   vagrant halt    → apaga la VM
#   vagrant destroy → elimina la VM

Vagrant.configure("2") do |config|

  config.vm.box = "ubuntu/jammy64"   # Ubuntu 22.04 LTS
  config.vm.hostname = "pneumoscan-deploy"

  # Recursos de la VM
  config.vm.provider "virtualbox" do |vb|
    vb.name   = "pneumoscan-deploy"
    vb.memory = 2048
    vb.cpus   = 2
  end

  # Montar el proyecto en /home/vagrant/app
  config.vm.synced_folder ".", "/home/vagrant/app",
    owner: "vagrant", group: "vagrant"

  # ── Provisioning ──────────────────────────────────────────────────
  config.vm.provision "shell", inline: <<-SHELL
    set -e
    export DEBIAN_FRONTEND=noninteractive

    echo "======================================"
    echo " Actualizando paquetes base..."
    echo "======================================"
    apt-get update -qq
    apt-get install -y -qq curl unzip git jq bash-completion

    # ── AWS CLI v2 ─────────────────────────────────────────────────
    echo "======================================"
    echo " Instalando AWS CLI v2..."
    echo "======================================"
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp/awscliv2
    /tmp/awscliv2/aws/install --update
    rm -rf /tmp/awscliv2 /tmp/awscliv2.zip
    aws --version

    # ── kubectl ────────────────────────────────────────────────────
    echo "======================================"
    echo " Instalando kubectl..."
    echo "======================================"
    KUBECTL_VERSION="v1.31.0"
    curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
      -o /usr/local/bin/kubectl
    chmod +x /usr/local/bin/kubectl
    kubectl version --client

    # ── eksctl ─────────────────────────────────────────────────────
    echo "======================================"
    echo " Instalando eksctl..."
    echo "======================================"
    curl -fsSL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz" \
      | tar -xz -C /usr/local/bin
    chmod +x /usr/local/bin/eksctl
    eksctl version

    # ── Docker ─────────────────────────────────────────────────────
    echo "======================================"
    echo " Instalando Docker..."
    echo "======================================"
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker vagrant

    # ── Autocompletion ─────────────────────────────────────────────
    echo "source /usr/share/bash-completion/bash_completion" >> /home/vagrant/.bashrc
    kubectl completion bash > /etc/bash_completion.d/kubectl
    eksctl completion bash > /etc/bash_completion.d/eksctl

    # ── Alias útiles ──────────────────────────────────────────────
    cat >> /home/vagrant/.bashrc << 'EOF'

# PneumoScan aliases
alias k="kubectl"
alias kgp="kubectl get pods -n pneumoscan"
alias kgs="kubectl get services -n pneumoscan"
alias kgd="kubectl get deployments -n pneumoscan"
alias klogs="kubectl logs -n pneumoscan"
cd /home/vagrant/app
EOF

    echo "======================================"
    echo " Provisioning completado."
    echo " Ejecuta: vagrant ssh"
    echo "======================================"
  SHELL

end
