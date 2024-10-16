/**
 *Submitted for verification at Etherscan.io on 2022-04-07
*/

// Litentry: Modified based on
// https://solidity-by-example.org/app/multi-sig-wallet/

// data:template
//
// (1) 0xcb10f215
    /**
        @notice Sets a new resource for handler contracts that use the IERCHandler interface,
        and maps the {handlerAddress} to {resourceID} in {_resourceIDToHandlerAddress}.
        @notice Only callable by an address that currently has the admin role.
        @param handlerAddress Address of handler resource will be set for.
        @param resourceID ResourceID to be used when making deposits.
        @param tokenAddress Address of contract to be called when a deposit is made and a deposited is executed.
     */
// Function: adminSetResource(address handlerAddress, bytes32 resourceID, address tokenAddress)
// MethodID: 0xcb10f215
// [0]:  00000000000000000000000050272b13efbb3da7c25cf5b98339efbd19a2a855
// [1]:  0000000000000000000000000000000000000000000000000000000000000001
// [2]:  00000000000000000000000027b981dd46ae0bfdda6677ddc75bce6995fca5bc
//
//
//
// (2) 0x4e056005
    /**
        @notice Modifies the number of votes required for a proposal to be considered passed.
        @notice Only callable by an address that currently has the admin role.
        @param newThreshold Value {_relayerThreshold} will be changed to.
        @notice Emits {RelayerThresholdChanged} event.
     */
// Function: adminChangeRelayerThreshold(uint256 newThreshold)
// MethodID: 0x4e056005
// [0]:  0000000000000000000000000000000000000000000000000000000000000001
//
//
//
// (3) 0xcdb0f73a
    /**
        @notice Grants {relayerAddress} the relayer role and increases {_totalRelayer} count.
        @notice Only callable by an address that currently has the admin role.
        @param relayerAddress Address of relayer to be added.
        @notice Emits {RelayerAdded} event.
     */
// Function: adminAddRelayer(address relayerAddress)
// MethodID: 0xcdb0f73a
// [0]:  0000000000000000000000002aa87a1dd75df16a6b22dd1d94ae6c3d3be40e88
//
//
//
// (4) 0x9d82dd63
    /**
        @notice Removes relayer role for {relayerAddress} and decreases {_totalRelayer} count.
        @notice Only callable by an address that currently has the admin role.
        @param relayerAddress Address of relayer to be removed.
        @notice Emits {RelayerRemoved} event.
     */
// Function: adminRemoveRelayer(address relayerAddress)
// MethodID: 0x9d82dd63
// [0]:  0000000000000000000000002aa87a1dd75df16a6b22dd1d94ae6c3d3be40e88
//
//
//
// (5) 0x80ae1c28
    /**
        @notice Pauses deposits, proposal creation and voting, and deposit executions.
        @notice Only callable by an address that currently has the admin role.
     */
// Function adminPauseTransfers()
// MethodID: 0x80ae1c28
//
//
//
// (6) 0xffaac0eb
    /**
        @notice Unpauses deposits, proposal creation and voting, and deposit executions.
        @notice Only callable by an address that currently has the admin role.
     */
// Function adminUnpauseTransfers()
// MethodID: 0xffaac0eb
//
//
//
// (7) 0x5e1fab0f
    /**
        @notice Removes admin role from {msg.sender} and grants it to {newAdmin}.
        @notice Only callable by an address that currently has the admin role.
        @param newAdmin Address that admin role will be granted to.
     */
// Function renounceAdmin(address newAdmin)
// MethodID: 0x5e1fab0f
// [0]:  0000000000000000000000002aa87a1dd75df16a6b22dd1d94ae6c3d3be40e88
//
//
//
// (8) 0x17f03ce5
    /**
        @notice Executes a deposit proposal that is considered passed using a specified handler contract.
        @notice Only callable by relayers when Bridge is not paused.
        @param chainID ID of chain deposit originated from.
        @param depositNonce ID of deposited generated by origin Bridge contract.
        @param dataHash Hash of data originally provided when deposit was made.
        @notice Proposal must be past expiry threshold.
        @notice Emits {ProposalEvent} event with status {Cancelled}.
     */
// Function cancelProposal(uint8 chainID, uint64 depositNonce, bytes32 dataHash)
// MethodID: 0x17f03ce5
// [0]:  0000000000000000000000000000000000000000000000000000000000000003
// [1]:  0000000000000000000000000000000000000000000000000000000000000007
// [2]:  00000000000000000000000000000063a7e2be78898ba83824b0c0cc8dfb6001
//
//
//
// (9) 0xc2d0c12d
    /**
        @notice Transfers eth in the contract to the specified addresses. The parameters addrs and amounts are mapped 1-1.
        This means that the address at index 0 for addrs will receive the amount (in WEI) from amounts at index 0.
        @param addrs Array of addresses to transfer {amounts} to.
        @param amounts Array of amonuts to transfer to {addrs}.
     */
// Function transferFunds(address payable[] calldata addrs, uint[] calldata amounts)
// MethodID: 0xc2d0c12d
// Too complicated. See official document for reference
// https://docs.soliditylang.org/en/develop/abi-spec.html#use-of-dynamic-types
//
//
//
// (10) 0x780cf004
    /**
        @notice Used to manually withdraw funds from ERC safes.
        @param handlerAddress Address of handler to withdraw from.
        @param tokenAddress Address of token to withdraw.
        @param recipient Address to withdraw tokens to.
        @param amountOrTokenID Either the amount of ERC20 tokens or the ERC721 token ID to withdraw.
     */
// Function adminWithdraw(address handlerAddress, address tokenAddress, address recipient, uint256 amountOrTokenID)
// MethodID: 0x780cf004
// [0]:  00000000000000000000000050272b13efbb3da7c25cf5b98339efbd19a2a855
// [1]:  00000000000000000000000027b981dd46ae0bfdda6677ddc75bce6995fca5bc
// [2]:  0000000000000000000000002aa87a1dd75df16a6b22dd1d94ae6c3d3be40e88
// [3]:  00000000000000000000000000000000000000000000000053444835ec580000
//
//
//
// (11) 0x7f3e3744
    /**
        @notice Changes deposit fee.
        @notice Only callable by admin.
        @param newFee Value {_fee} will be updated to.
     */
// Function adminChangeFee(uint newFee)
// MethodID: 0x7f3e3744
// [0]:  00000000000000000000000000000000000000000000000053444835ec580000
//
//
//
// (12) 0x8c0c2631
    /**
        @notice Sets a resource as burnable for handler contracts that use the IERCHandler interface.
        @notice Only callable by an address that currently has the admin role.
        @param handlerAddress Address of handler resource will be set for.
        @param tokenAddress Address of contract to be called when a deposit is made and a deposited is executed.
     */
// Function adminSetBurnable(address handlerAddress, address tokenAddress)
// MethodID: 0x8c0c2631
// [0]:  00000000000000000000000050272b13efbb3da7c25cf5b98339efbd19a2a855
// [1]:  00000000000000000000000027b981dd46ae0bfdda6677ddc75bce6995fca5bc
//
//
//
// (13) 0xe8437ee7
    /**
        @notice Sets a new resource for handler contracts that use the IGenericHandler interface,
        and maps the {handlerAddress} to {resourceID} in {_resourceIDToHandlerAddress}.
        @notice Only callable by an address that currently has the admin role.
        @param handlerAddress Address of handler resource will be set for.
        @param resourceID ResourceID to be used when making deposits.
        @param contractAddress Address of contract to be called when a deposit is made and a deposited is executed.
     */
// Function adminSetGenericResource(address handlerAddress, bytes32 resourceID, address contractAddress, bytes4 depositFunctionSig, bytes4 executeFunctionSig)
// MethodID: 0xe8437ee7
// [0]:  00000000000000000000000050272b13efbb3da7c25cf5b98339efbd19a2a855
// [1]:  00000000000000000000000000000063a7e2be78898ba83824b0c0cc8dfb6001
// [2]:  00000000000000000000000027b981dd46ae0bfdda6677ddc75bce6995fca5bc
// [3]:  ****************************************************************
// [4]:  ****************************************************************


// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

contract MultiSigWallet {
    event Deposit(address indexed sender, uint amount, uint balance);
    event SubmitTransaction(
        address indexed owner,
        uint indexed txIndex,
        address indexed to,
        uint value,
        bytes data
    );
    event ConfirmTransaction(address indexed owner, uint indexed txIndex);
    event RevokeConfirmation(address indexed owner, uint indexed txIndex);
    event ExecuteTransaction(address indexed owner, uint indexed txIndex);
    event addProposer(address indexed proposer,address indexed owner);
    event removeProposer(address indexed proposer, address indexed owner);

    address[] public owners;
    mapping(address => bool) public isOwner;
    mapping(address => bool) public isProposer;
    uint public numConfirmationsRequired;

    struct Transaction {
        address to;
        uint value;
        bytes data;
        bool executed;
        uint numConfirmations;
    }

    // mapping from tx index => owner => bool
    mapping(uint => mapping(address => bool)) public isConfirmed;

    Transaction[] public transactions;

    modifier onlyOwner() {
        require(isOwner[msg.sender], "not owner");
        _;
    }

    modifier txExists(uint _txIndex) {
        require(_txIndex < transactions.length, "tx does not exist");
        _;
    }

    modifier notExecuted(uint _txIndex) {
        require(!transactions[_txIndex].executed, "tx already executed");
        _;
    }

    modifier notConfirmed(uint _txIndex) {
        require(!isConfirmed[_txIndex][msg.sender], "tx already confirmed");
        _;
    }

    constructor(address[] memory _owners, uint _numConfirmationsRequired) {
        require(_owners.length > 0, "owners required");
        require(
            _numConfirmationsRequired > 0 &&
                _numConfirmationsRequired <= _owners.length,
            "invalid number of required confirmations"
        );

        for (uint i = 0; i < _owners.length; i++) {
            address owner = _owners[i];

            require(owner != address(0), "invalid owner");
            require(!isOwner[owner], "owner not unique");

            isOwner[owner] = true;
            owners.push(owner);
        }

        numConfirmationsRequired = _numConfirmationsRequired;
    }

    receive() external payable {
        emit Deposit(msg.sender, msg.value, address(this).balance);
    }

    function addProposers(address[] calldata _proposers) public onlyOwner {
        for (uint i = 0; i < _proposers.length; i++) {
            require(!isProposer[_proposers[i]], "proposer already");
            isProposer[_proposers[i]] = true;
            emit addProposer(_proposers[i], msg.sender);
        }
    }
    
    function removeProposers(address[] calldata _proposers) public onlyOwner {
        for (uint i = 0; i < _proposers.length; i++) {
            require(isProposer[_proposers[i]], "not proposer already");
            isProposer[_proposers[i]] = false;
            emit removeProposer(_proposers[i], msg.sender);
        }
    }

    function submitTransaction(
        address _to,
        uint _value,
        bytes memory _data
    ) public {
        require(isOwner[msg.sender] || isProposer[msg.sender], "not owner/proposer");


        uint txIndex = transactions.length;

        transactions.push(
            Transaction({
                to: _to,
                value: _value,
                data: _data,
                executed: false,
                numConfirmations: 0
            })
        );

        emit SubmitTransaction(msg.sender, txIndex, _to, _value, _data);
    }

    function confirmTransaction(uint _txIndex)
        public
        onlyOwner
        txExists(_txIndex)
        notExecuted(_txIndex)
        notConfirmed(_txIndex)
    {
        Transaction storage transaction = transactions[_txIndex];
        transaction.numConfirmations += 1;
        isConfirmed[_txIndex][msg.sender] = true;

        emit ConfirmTransaction(msg.sender, _txIndex);
    }

    function executeTransaction(uint _txIndex)
        public
        onlyOwner
        txExists(_txIndex)
        notExecuted(_txIndex)
    {
        Transaction storage transaction = transactions[_txIndex];

        require(
            transaction.numConfirmations >= numConfirmationsRequired,
            "cannot execute tx"
        );

        transaction.executed = true;

        (bool success, ) = transaction.to.call{value: transaction.value}(
            transaction.data
        );
        require(success, "tx failed");

        emit ExecuteTransaction(msg.sender, _txIndex);
    }

    function revokeConfirmation(uint _txIndex)
        public
        onlyOwner
        txExists(_txIndex)
        notExecuted(_txIndex)
    {
        Transaction storage transaction = transactions[_txIndex];

        require(isConfirmed[_txIndex][msg.sender], "tx not confirmed");

        transaction.numConfirmations -= 1;
        isConfirmed[_txIndex][msg.sender] = false;

        emit RevokeConfirmation(msg.sender, _txIndex);
    }

    function getOwners() public view returns (address[] memory) {
        return owners;
    }

    function getTransactionCount() public view returns (uint) {
        return transactions.length;
    }

    function getTransaction(uint _txIndex)
        public
        view
        returns (
            address to,
            uint value,
            bytes memory data,
            bool executed,
            uint numConfirmations
        )
    {
        Transaction storage transaction = transactions[_txIndex];

        return (
            transaction.to,
            transaction.value,
            transaction.data,
            transaction.executed,
            transaction.numConfirmations
        );
    }
}