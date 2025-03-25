import hashlib
import json
import time

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash="0")  # Genesis block
    
    def create_block(self, data="", previous_hash=None):
        """Creates a new block with AI-generated inventory data."""
        if previous_hash is None and len(self.chain) > 0:
            previous_hash = self.get_latest_block()["hash"]
        elif previous_hash is None:
            previous_hash = "0"
            
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time.time(),
            "data": data,
            "previous_hash": previous_hash,
            "hash": ""
        }
        block["hash"] = self.hash(block)
        self.chain.append(block)
        return block
    
    def get_latest_block(self):
        """Returns the latest block in the blockchain."""
        if len(self.chain) == 0:
            return None
        return self.chain[-1]
    
    def hash(self, block):
        """Generates a SHA-256 hash for the block."""
        # Create a copy without the hash field to avoid circular reference
        block_copy = block.copy()
        block_copy.pop("hash", None)
        encoded_block = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
    
    def get_chain(self):
        """Returns the entire blockchain."""
        return self.chain

# Initialize Blockchain
blockchain = Blockchain()